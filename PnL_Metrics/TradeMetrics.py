import numpy as np
import pandas as pd


class TradingMetrics:
    def __init__(self, trade_pnls):
        """
        Initialize the trade‐level metrics calculator.

        Parameters:
        - trade_pnls: pd.DataFrame or pd.Series containing P&L for each closed trade
        """
        if isinstance(trade_pnls, pd.Series):
            self.trade_pnls = trade_pnls.astype(float)
        else:
            self.trade_pnls = pd.Series(trade_pnls, dtype=float)

    def win_rate(self):
        """
        Calculate win rate aka fraction of trades with positive P&L.
        """
        if self.trade_pnls.empty:
            return np.nan
        wins = (self.trade_pnls > 0).sum()
        return wins / len(self.trade_pnls)

    def loss_rate(self):
        """
        Calculate loss rate aka fraction of trades with negative P&L.
        """
        if self.trade_pnls.empty:
            return np.nan
        losses = (self.trade_pnls < 0).sum()
        return losses / len(self.trade_pnls)

    def average_win(self):
        """
        Calculate average win.
        """
        wins = self.trade_pnls[self.trade_pnls > 0]
        return wins.mean() if not wins.empty else 0.0

    def average_loss(self):
        """
        Calculate average loss.
        """
        losses = self.trade_pnls[self.trade_pnls < 0]
        return losses.mean() if not losses.empty else 0.0

    def expectancy(self):
        """
        Calculate trade expectancy aka expected P&L per trade.
        """
        if self.trade_pnls.empty:
            return np.nan
        w = self.win_rate()
        aw = self.average_win()
        al = self.average_loss()
        return w * aw + (1 - w) * al

    def summary(self):
        """
        Generate a summary of all trade‐level metrics.
        """
        all_metrics = {
            'Win Rate':        self.win_rate(),
            'Loss Rate':       self.loss_rate(),
            'Average Win':     self.average_win(),
            'Average Loss':    self.average_loss(),
            'Expectancy':      self.expectancy(),
        }
        
        summary_df = pd.DataFrame(all_metrics, index=['TradingMetrics'])
        return summary_df

if __name__ == "__main__":
    # Define portfolio
    symbol_manager = BinanceSymbolManager()
    price_fetcher = BinancePriceFetcher(symbol_manager)

    # Add symbols
    print(symbol_manager.add_symbol("BTCUSDT"))  # Success
    print(symbol_manager.add_symbol("ETHUSDT"))  # Success

    btc_portfolio_daily = price_fetcher.get_grp_historical_ohlcv(
        interval="1d",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    weights = np.array([0.5, 0.5])  # Equal-weighted

    # Initialize strategy
    strategy = SimpleStrategy(
        tickers=symbol_manager.get_symbols(),
        weights=weights,
        data=btc_portfolio_daily
    )

    returns = strategy.momentum_strategy(lookback=90, hold_period=30)
    returns["Strat2"] = returns

    # Calculate P&L
    daily_returns = returns.iloc[:, 0]
    trade_pnls = []
    current_pnl = 0.0
    in_trade = False
    
    for r in daily_returns:
        if r != 0:
            in_trade = True
            current_pnl += r
        else:
            if in_trade:
                trade_pnls.append(current_pnl)
                current_pnl = 0.0
                in_trade = False
    if in_trade:
        trade_pnls.append(current_pnl)
    
    # Initialize metrics calculator
    trading_metrics = TradingMetrics(trade_pnls)

    # Get individual metrics
    print("Win Rate:\n", trading_metrics.win_rate())
    print("\nLoss Rate:\n", trading_metrics.loss_rate())
    print("\nAverage Win:\n", trading_metrics.average_win())
    print("\nAverage Loss:\n", trading_metrics.average_loss())
    print("\nExpectancy:\n", trading_metrics.expectancy())

    # Get full summary
    print("\nTrading Metrics Summary:")
    summary = trading_metrics.summary()
    print(summary)