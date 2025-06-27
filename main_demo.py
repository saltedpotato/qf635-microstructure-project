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

from Data.BinancePriceFetcher import *
from Data.GetOrderBook import *
from Utils.config import *
from Strats.MeanReversionStrat import *
from Strats.RegressionStrat import *
from Strats.RSI import *

# === Start API Service ===
app = Flask(__name__)
CORS(app, resources={r"/signals": {"origins": "*"}})
_latest_signals = None
_lock = threading.Lock()

@app.route("/signals")
def signals_endpoint():
    with _lock:
        if _latest_signals is None or _latest_signals.empty:
            return jsonify([])
        return jsonify(_latest_signals.to_dict(orient="records"))

def _run_flask():
    app.run(host="0.0.0.0", port=8888, threaded=False, debug=False)

threading.Thread(target=_run_flask, daemon=True).start()
# === End API Service ===

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
FEE_RATE = 0.001 # 0.1% quoted from binance
ui_interval_seconds = 10  # UI update every 30 seconds
signal_refresh_interval = interval_mins * 60  # Signal every 5 minutes
last_signal_time = 0

def get_signal(prices, portfolio_prices):
    signals = pd.DataFrame()

    for ind, t in enumerate(passed_tickers):
        mid_price = prices[ind][1]  # prices[ind] = [bid, mid, ask]
        
        # Dummy BUY signal
        signal = pd.DataFrame([{
            'timestamp': datetime.now(),
            'Tickers': t,
            'signals': 1,         # Force BUY
            'weights': 1.0,       # Equal weight for simplicity
            'exit_signals': 0,    # No exit
            'Price': mid_price
        }])
        signals = pd.concat([signals, signal])

    # Normalize weights
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

        now = time.time()
        if now - last_signal_time >= signal_refresh_interval:
            signals = get_signal(live_prices, portfolio_prices)
            with _lock:
                _latest_signals = signals.copy()
            last_signal_time = now

        # === Update capital, position, and realised PnL ===
        for s in passed_tickers:
            if _latest_signals is None or s not in _latest_signals['Tickers'].values:
                continue
            row = _latest_signals[_latest_signals['Tickers'] == s].iloc[0]
            sig = int(row['signals'])
            exit_sig = int(row['exit_signals'])

            if sig == 1:
                qty = (row['weights'] * capital_per_round) / row['Price']
                cost = qty * row['Price'] * (1 + FEE_RATE)
                if capital >= cost:
                    capital -= cost
                    current_position[s] += qty
                    entry_prices[s] = row['Price'] 

            elif sig == -1 and current_position[s] > 0:
                qty = min(current_position[s], (row['weights'] * capital_per_round) / row['Price'])
                proceeds = qty * row['Price'] * (1 - FEE_RATE)
                realised_pnl += qty * (row['Price'] - entry_prices[s])  # compute realised PnL
                capital += proceeds
                current_position[s] -= qty
                if current_position[s] <= 0:
                    entry_prices[s] = None  
            elif exit_sig == 1 and current_position[s] > 0:
                qty = current_position[s]
                proceeds = qty * row['Price'] * (1 - FEE_RATE)
                realised_pnl += qty * (row['Price'] - entry_prices[s])  # compute realised PnL
                capital += proceeds
                current_position[s] = 0
                entry_prices[s] = None  


        unrealised_pnl = sum(
            (live_prices[i][1] - entry_prices[s]) * current_position[s]
            for i, s in enumerate(passed_tickers)
            if current_position[s] > 0 and entry_prices[s] is not None
        )
        position_value = sum(
            live_prices[i][1] * current_position[s]
            for i, s in enumerate(passed_tickers)
            if current_position[s] > 0
        )
        portfolio_value = capital + position_value


        print("\n====== LIVE TRADING VIEW ======")
        print(f"Portfolio Value : ${portfolio_value:,.2f}")
        print(f"Realised PnL    : ${realised_pnl:,.2f}")
        print(f"Unrealised PnL  : ${unrealised_pnl:,.2f}")
        print(f"Capital         : ${capital:,.2f}")
        for s in passed_tickers:
            if _latest_signals is None or s not in _latest_signals['Tickers'].values:
                continue
            row = _latest_signals[_latest_signals['Tickers'] == s].iloc[0]
            sig = int(row['signals'])
            exit_sig = int(row['exit_signals'])
            sig_label = 'BUY' if sig == 1 else ('SELL' if sig == -1 else 'HOLD')
            qty = 0.0
            if sig == 1:
                qty = (row['weights'] * capital_per_round) / row['Price']
                print(f"{s} SIGNAL: {sig_label} {qty:.4f} Qty @ {row['Price']:.2f} | Current Position: {current_position[s]:.2f}")
            elif sig == -1 and current_position[s] > 0:
                qty = min(current_position[s], (row['weights'] * capital_per_round) / row['Price'])
                print(f"{s} SIGNAL: {sig_label} {qty:.4f} Qty @ {row['Price']:.2f} | Current Position: {current_position[s]:.2f}")
            elif exit_sig == 1:
                qty = current_position[s]
                print(f"{s} SIGNAL: EXIT {qty:.4f} Qty @ {row['Price']:.2f} | Current Position: {current_position[s]:.2f}")
            else:
                print(f"{s} SIGNAL: HOLD @ {row['Price']:.2f} | Current Position: {current_position[s]:.2f}")
        print("==============================\n")

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