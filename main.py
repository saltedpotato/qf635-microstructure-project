from pathlib import Path
import time
import logging
from datetime import datetime, timezone
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from Data.GetOrderBook import *
from Strats.MeanReversionStrat import *
from Strats.RegressionStrat import *
from Strats.RSI import *

# === Start API Service ===
app = Flask(__name__)
CORS(app, resources={r"/signals": {"origins": "*"}})
_latest_signals = None
_latest_json = []

@app.route("/signals")
def signals_endpoint():
    return jsonify(_latest_json or {"summary": {}, "signals": []})

def _run_flask():
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8888, threaded=False, debug=False)

threading.Thread(target=_run_flask, daemon=True).start()

print("Trading parameters (edit in config.py file):")
print(f"Tickers: {tickers}")
print(f"Stop loss: {stoploss}")
print(f"Max drawdown duration: {drawdown_duration}")
print(f"Rolling window: {rolling}")
print(f"Portfolio Allocation method: {weight_method.__name__}")
print(f"Allow shorting in weight allocation: {short}")
print(f"Train start date: {start_date}")
print(f"Trading Frequency: {interval}")

symbol_manager = BinanceSymbolManager()
for t in tickers:
    symbol_manager.add_symbol(t)

passed_tickers = symbol_manager.get_symbols()
price_fetcher = BinancePriceFetcher(passed_tickers)
order_book_manager = OrderBookManager(passed_tickers)

print("Fetching data...")
portfolio_prices = price_fetcher.get_grp_historical_ohlcv(
    interval=interval,
    start_date=start_date,
    end_date=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
)
print("Fetched data...")
portfolio_prices.to_csv("./portfolio_prices.csv", index=False)

initial_capital = 100_000
capital = initial_capital
current_position = {symbol: 0 for symbol in passed_tickers}
entry_prices = {symbol: None for symbol in passed_tickers}
realised_pnl = 0
capital_per_round = 1000
FEE_RATE = 0.001
ui_interval_seconds = 5
signal_refresh_interval = interval_mins * 60
last_signal_time = 0

def get_signal(prices, portfolio_prices):
    signals = pd.DataFrame()

    mr_signals = {
        'mr': MeanReversionStrat(portfolio_prices[-MeanReversionStrat_PARAMS.lookback:].copy(), passed_tickers),
        'rsi': RSI(portfolio_prices[-RegressionStrat_PARAMS.lookback_window - 10:].copy(), passed_tickers)
    }

    trend_signals = {
        'lr': RegressionStrat(
            portfolio_prices[-RegressionStrat_PARAMS.lookback_window - 10:].copy(), passed_tickers,
            lookback_window=RegressionStrat_PARAMS.lookback_window,
            regression_type=RegressionStrat_PARAMS.regression_type)
    }

    for ind, t in enumerate(passed_tickers):
        huber_result = hurst_exponent(portfolio_prices[t].values[-rolling:])
        if huber_result < Hurst_Type.mean_revert[-1]:
            mr_signal = mr_signals['mr'].generate_single_signal(t, prices[ind],
                lookback=MeanReversionStrat_PARAMS.lookback,
                execute_threshold=MeanReversionStrat_PARAMS.execute_threshold,
                close_threshold=MeanReversionStrat_PARAMS.close_threshold)

            rsi_signal = mr_signals['rsi'].generate_single_signal(t, prices[ind],
                rsi_period=RSI_PARAMS.rsi_period,
                stoch_period=RSI_PARAMS.stoch_period,
                k_smooth=RSI_PARAMS.k_smooth,
                d_smooth=RSI_PARAMS.d_smooth)

            voted_signal = (mr_signal['signals'].iloc[0] + rsi_signal['signals'].iloc[0]) / 2
            if voted_signal <= -0.5:
                voted_signal = -1
            elif voted_signal >= 0.5:
                voted_signal = 1
            else:
                voted_signal = 0

            voted_weights = (mr_signal['weights'].iloc[0] + rsi_signal['weights'].iloc[0]) / 2
            voted_exit_signal = (mr_signal['exit_signals'].iloc[0] + rsi_signal['exit_signals'].iloc[0]) / 2
            voted_exit_signal = 1 if voted_exit_signal > 0.5 else 0

            signal = mr_signal.copy()
            signal['signals'] = voted_signal
            signal['weights'] = voted_weights
            signal['exit_signals'] = voted_exit_signal
        else:
            signal = trend_signals['lr'].generate_single_signal(
                t, prices[ind],
                pca_components=RegressionStrat_PARAMS.pca_components,
                execute_threshold=RegressionStrat_PARAMS.execute_threshold,
                r2_exit=RegressionStrat_PARAMS.r2_exit)

        signals = pd.concat([signals, signal])

    signals['timestamp'] = datetime.now()
    signals = signals[['timestamp', 'Tickers', 'signals', 'weights', 'exit_signals', 'Price']]
    weight_sum = signals['weights'].sum()
    signals['weights'] = signals['weights'] / weight_sum if weight_sum > 0 else 0
    return signals

print("\n====== LIVE TRADING STARTED ======")
print(f"Initial Capital  : ${initial_capital:,.2f}")
print(f"Monitoring Tickers: {', '.join(passed_tickers)}\n")

while True:
    try:
        start_time = time.time()
        live_prices = []
        for t in passed_tickers:
            ob = order_book_manager.fetch_order_book(t, limit=5)
            live_prices.append([ob.best_bid(), ob.get_mid_price(), ob.best_ask()])

        unrealised_pnl = 0
        for i, s in enumerate(passed_tickers):
            if current_position[s] != 0 and entry_prices[s] is not None:
                entry = entry_prices[s]
                current = live_prices[i][1]
                if current_position[s] > 0:
                    unrealised_pnl += (current - entry) * current_position[s]
                else:
                    unrealised_pnl += (entry - current) * abs(current_position[s])
        unrealised_pnl_total = unrealised_pnl

        now = time.time()
        if now - last_signal_time >= signal_refresh_interval:
            signals = get_signal(live_prices, portfolio_prices)
            _latest_signals = signals.copy()
            last_signal_time = now

        for s in passed_tickers:
            if _latest_signals is None or s not in _latest_signals['Tickers'].values:
                continue
            row = _latest_signals[_latest_signals['Tickers'] == s].iloc[0]
            sig = int(row['signals'])
            exit_sig = int(row['exit_signals'])
            price = row['Price']
            weight = row['weights']

            qty = (weight * capital_per_round) / price

            if sig == 1:
                cost = qty * price * (1 + FEE_RATE)
                if capital >= cost:
                    capital -= cost
                    current_position[s] += qty
                    entry_prices[s] = price

            elif sig == -1:
                proceeds = qty * price * (1 - FEE_RATE)
                capital += proceeds
                if current_position[s] >= 0:
                    entry_prices[s] = price
                if current_position[s] < 0:
                    realised_pnl += -qty * (price - entry_prices[s])
                current_position[s] -= qty

            elif exit_sig == 1 and current_position[s] != 0:
                qty = abs(current_position[s])
                exit_price = price
                entry_price = entry_prices[s] or exit_price

                if current_position[s] > 0:
                    proceeds = qty * exit_price * (1 - FEE_RATE)
                    realised_pnl += qty * (exit_price - entry_price)
                    capital += proceeds
                else:
                    cost_to_close = qty * exit_price * (1 + FEE_RATE)
                    realised_pnl += qty * (entry_price - exit_price)
                    capital -= cost_to_close

                current_position[s] = 0
                entry_prices[s] = None

        position_value = sum(
            live_prices[i][1] * current_position[s]
            for i, s in enumerate(passed_tickers)
        )
        portfolio_value = capital + position_value

        signals_json = []
        print("\n====== LIVE TRADING VIEW ======")
        print(f"Portfolio Value : ${portfolio_value:,.2f}")
        print(f"Realised PnL    : ${realised_pnl:,.2f}")
        print(f"Unrealised PnL  : ${unrealised_pnl_total:,.2f}")
        print(f"Capital         : ${capital:,.2f}")

        for s in passed_tickers:
            if _latest_signals is None or s not in _latest_signals['Tickers'].values:
                continue
            row = _latest_signals[_latest_signals['Tickers'] == s].iloc[0]
            sig = int(row['signals'])
            exit_sig = int(row['exit_signals'])
            sig_label = 'BUY' if sig == 1 else ('SELL' if sig == -1 else 'HOLD')
            qty = (row['weights'] * capital_per_round) / row['Price']
            mid_price = next((p[1] for i, p in enumerate(live_prices) if passed_tickers[i] == s), None)

            print(f"{s} SIGNAL: {sig_label} {qty:.4f} Qty @ {mid_price:.2f} | Current Position: {current_position[s]:.2f}")
            signals_json.append({
                "ticker": s,
                "signal": sig_label,
                "exit_signal": exit_sig,
                "price": f"{round(mid_price, 2):.2f}",
                "current_position": current_position[s],
                "qty": f"{round(qty, 2):.2f}",
                "qty_price": f"{row['Price']:.2f}",
            })

        print("==============================\n")

        summary = {
            "portfolio_value": portfolio_value,
            "realised_pnl": realised_pnl,
            "unrealised_pnl": unrealised_pnl_total,
            "capital": capital,
        }

        _latest_json = {
            "summary": summary,
            "signals": signals_json,
        }

        portfolio_prices_start = time.perf_counter()
        portfolio_prices = price_fetcher.get_grp_historical_ohlcv(
            interval=interval,
            start_date=start_date,
            end_date=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        )
        portfolio_prices_end = time.perf_counter()
        time.sleep(max(ui_interval_seconds - (portfolio_prices_end - portfolio_prices_start), 0))

    except KeyboardInterrupt:
        logging.info("Manual stop triggered.")
        break
    except Exception as e:
        logging.error(f"Error: {e}")
        time.sleep(5)

