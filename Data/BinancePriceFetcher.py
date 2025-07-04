from datetime import timezone
from pathlib import Path
from Data.BinanceSymbolManager import *

class BinancePriceFetcher:
    def __init__(self, symbols: list):
        """Initialize with a symbol manager instance."""
        self.symbols = symbols
        self.session = requests.Session()

    def get_price(self, symbol: str):
        """Fetch current price for a single symbol."""
        symbol = symbol.upper().strip()

        response = requests.get(
            f"{BASE_URL}/ticker/price",
            params={"symbol": symbol}
        )

        if response.status_code == 200:
            data = response.json()
            return {"symbol": symbol, "price": float(data["price"])}
        else:
            return {
                "symbol": symbol,
                "error": f"API Error {response.status_code}",
                "details": response.text
            }

    def get_all_prices(self):
        """Fetch prices for all tracked symbols."""
        results = {}
        for symbol in self.symbols:
            result = self.get_price(symbol)
            if "price" in result:
                results[symbol] = result["price"]
            else:
                results[symbol] = result["error"]
        return results

    def _parse_trades_to_dataframe(self, trades_data: list) -> pd.DataFrame:
        """
        Convert Binance historical trades data to pandas DataFrame.

        Args:
            trades_data: List of trade dictionaries from Binance API

        Returns:
            pd.DataFrame with columns:
            - timestamp (datetime)
            - price (float)
            - quantity (float)
            - quote_quantity (float)
            - is_buyer_maker (bool)
            - trade_id (int)
        """
        if not trades_data:
            return pd.DataFrame()

        # Create DataFrame from raw data
        df = pd.DataFrame(trades_data)

        # Convert data types
        df['price'] = df['price'].astype(float)
        df['qty'] = df['qty'].astype(float)
        df['quoteQty'] = df['quoteQty'].astype(float)
        df['isBuyerMaker'] = df['isBuyerMaker'].astype(bool)
        df['time'] = pd.to_datetime(df['time'], unit='ms')  # Convert ms timestamp to datetime

        # Rename columns to more standard names
        df = df.rename(columns={
            'time': 'timestamp',
            'qty': 'quantity',
            'quoteQty': 'quote_quantity',
            'isBuyerMaker': 'is_buyer_maker',
            'id': 'trade_id'
        })

        # Set timestamp as index
        df = df.set_index('timestamp')

        # Reorder columns
        return df[['trade_id', 'price', 'quantity', 'quote_quantity', 'is_buyer_maker']]

    def get_historical_trades(self, symbol: str, limit: int = 1000):
        """
        Get historical trades (executed bids/asks)
        Docs: https://binance-docs.github.io/apidocs/spot/en/#old-trade-lookup
        """
        url = "https://api.binance.com/api/v3/historicalTrades"
        params = {
            "symbol": symbol.upper().strip(),
            "limit": limit  # Max 1000
        }
        response = requests.get(url, params=params)
        trades_df = self._parse_trades_to_dataframe(response.json())
        return trades_df

    def get_klines(self, symbol: str, interval: str, start_time: str = None, end_time: str = None,
                   limit: int = 1000) -> pd.DataFrame:
        """
        Get historical OHLCV data from Binance

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval ('1m', '5m', '1h', '1d', etc.)
            start_time: Start time in YYYY-MM-DD format
            end_time: End time in YYYY-MM-DD format
            limit: Number of data points (max 1000)

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume, close_time, ...]
        """
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = int(pd.to_datetime(start_time).timestamp() * 1000)
        if end_time:
            params['endTime'] = int(pd.to_datetime(end_time).timestamp() * 1000)

        response = self.session.get(f"{BASE_URL}/klines", params=params)
        response.raise_for_status()

        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]

        df = pd.DataFrame(response.json(), columns=columns)

        # Convert types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def get_historical_ohlcv(self, symbol: str, interval: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Get complete historical OHLCV data between dates

        Args:
            symbol: Trading pair
            interval: Kline interval
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD, defaults to now)

        Returns:
            Concatenated DataFrame with all historical data
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        all_data = []
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        while current_date < end_date:
            # print(current_date)
            # Binance has 1000 data point limit per request
            data = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_date.strftime('%Y-%m-%d %H:%M:%S'),
                limit=1000,
            )

            if data.empty or data['timestamp'].iloc[-1] == current_date:
                break

            all_data.append(data)
            current_date = data['timestamp'].iloc[-1]
            time.sleep(0.1)  # Rate limit

        if all_data:
            final = pd.concat(all_data).drop_duplicates().reset_index(drop=True)
            final = final[final['timestamp']<= end_date]
            return final
        return pd.DataFrame()

    def get_grp_historical_ohlcv(self, interval: str, start_date: str, end_date: str = None, col: str = "close") -> pd.DataFrame:
        """
        Get complete historical OHLCV data between dates

        Args:
            interval: Kline interval
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD, defaults to now)

        Returns:
            Concatenated DataFrame with all historical data
        """
        if Path("./portfolio_prices.csv").exists():
            compiled_symbol_df = pd.read_csv("./portfolio_prices.csv", parse_dates=["timestamp"])
            last_ts = compiled_symbol_df["timestamp"].max()
            next_ts = last_ts + pd.Timedelta(minutes=5)
        else:
            compiled_symbol_df = pd.DataFrame()
            next_ts = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S')

        df_new = pd.DataFrame()
        for s in self.symbols:
            symbol_df = self.get_historical_ohlcv(s, interval, next_ts, end_date)

            if symbol_df.empty:
                break

            symbol_df = symbol_df[["timestamp", col]].copy()
            symbol_df = symbol_df.rename(columns={col:s})

            if len(df_new) == 0:
                df_new = symbol_df
            else:
                df_new = df_new.merge(symbol_df, on="timestamp", how="left")

        if df_new.empty:
            return compiled_symbol_df

        if compiled_symbol_df.empty:
            compiled_symbol_df = df_new
        else:
            compiled_symbol_df.merge(df_new)

        return compiled_symbol_df



# Test
if __name__ == "__main__":
    # Initialize managers
    symbol_manager = BinanceSymbolManager()

    # Add symbols
    print(symbol_manager.add_symbol("BTCUSDT"))  # Success
    print(symbol_manager.add_symbol("ETHUSDT"))  # Success
    print(symbol_manager.add_symbol("INVALID"))  # Will not add

    price_fetcher = BinancePriceFetcher(symbol_manager.get_symbols())
    # Fetch single price
    print("\nSingle Price Fetch:")
    print(price_fetcher.get_price("btcusdt"))  # {'symbol': 'BTCUSDT', 'price': 61234.50}
    print(price_fetcher.get_price("XRPUSDT"))  # {'symbol': 'XRPUSDT', 'price': 2.3845}

    # Fetch all prices
    print("\nAll Prices:")
    prices = price_fetcher.get_all_prices()
    for symbol, price in prices.items():
        print(f"{symbol}: {price}")

    # Return historical price
    print("\nHistorical Price:")
    print(price_fetcher.get_historical_trades("BTCUSDT", limit=100))

    # Get daily BTCUSDT data for 2023
    btc_daily = price_fetcher.get_historical_ohlcv(
        symbol="BTCUSDT",
        interval="1d",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    btc_portfolio = price_fetcher.get_grp_historical_ohlcv(
        interval="5m",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

    print(btc_daily.head())
    print(btc_portfolio.head())
