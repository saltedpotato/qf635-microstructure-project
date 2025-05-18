import requests
from get_clean_data.BinanceSymbolManager import *

class BinancePriceFetcher:
    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self, symbol_manager: BinanceSymbolManager):
        """Initialize with a symbol manager instance."""
        self.symbol_manager = symbol_manager

    def get_price(self, symbol: str):
        """Fetch current price for a single symbol."""
        symbol = symbol.upper().strip()
        if not self.symbol_manager.has_symbol(symbol):
            return {"error": f"Symbol '{symbol}' not being tracked"}

        response = requests.get(
            f"{self.BASE_URL}/ticker/price",
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
        for symbol in self.symbol_manager.get_symbols():
            result = self.get_price(symbol)
            if "price" in result:
                results[symbol] = result["price"]
            else:
                results[symbol] = result["error"]
        return results

    def get_order_book(self, symbol: str, limit=10):
        '''
        Returns Price, quantity
        Bids are sorted in descending price order (best/highest bid first).
        Asks are sorted in ascending price order (best/lowest ask first).
        '''

        url = self.BASE_URL+"/depth"
        symbol = symbol.upper().strip()
        params = {
            "symbol": symbol,
            "limit": limit  # Number of orders to fetch
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"Order Book for {symbol}:")
            print("Bids (Buy Orders):", data['bids'][:5])  # Top 5 bids
            print("Asks (Sell Orders):", data['asks'][:5])  # Top 5 asks
        else:
            print(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    # Initialize managers
    symbol_manager = BinanceSymbolManager()
    price_fetcher = BinancePriceFetcher(symbol_manager)

    # Add symbols
    print(symbol_manager.add_symbol("BTCUSDT"))  # Success
    print(symbol_manager.add_symbol("ETHUSDT"))  # Success
    print(symbol_manager.add_symbol("INVALID"))  # Will add but fail in API

    # Fetch single price
    print("\nSingle Price Fetch:")
    print(price_fetcher.get_price("btcusdt"))  # {'symbol': 'BTCUSDT', 'price': 61234.50}
    print(price_fetcher.get_price("XRPUSDT"))  # Error - not tracked

    # Fetch all prices
    print("\nAll Prices:")
    prices = price_fetcher.get_all_prices()
    for symbol, price in prices.items():
        print(f"{symbol}: {price}")

    print("\nSingle Price Order Book Fetch:")
    print(price_fetcher.get_order_book("btcusdt", limit=20))  # {'symbol': 'BTCUSDT', 'price': 61234.50}
    print(price_fetcher.get_price("XRPUSDT"))  # Error - not tracked

    # Remove a symbol and re-fetch
    print("\nAfter Removal:")
    print(symbol_manager.remove_symbol("ETHUSDT"))
    print(price_fetcher.get_all_prices())