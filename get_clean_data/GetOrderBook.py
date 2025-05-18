from get_clean_data.import_files import *
from get_clean_data.BinanceSymbolManager import *

# Tier represents a price level
class Tier:
    def __init__(self, price: float, size: float, quote_id: str = None):
        self.price = price
        self.size = size
        self.quote_id = quote_id


# OrderBook class to hold bids and asks, as an array of Tiers
class OrderBook:
    def __init__(self, _timestamp: float, _bids: [Tier], _asks: [Tier], symbol: str):
        self.timestamp = _timestamp
        self.bids = _bids
        self.asks = _asks
        self.symbol = symbol

    # method to get best bid
    def best_bid(self):
        return self.bids[0].price if self.bids else None

    # method to get best ask
    def best_ask(self):
        return self.asks[0].price if self.asks else None

    def get_mid_price(self):
        if self.best_bid() and self.best_ask():
            return (self.best_bid() + self.best_ask()) / 2
        return None

    def __str__(self):
        return f"{self.symbol} @ {datetime.fromtimestamp(self.timestamp)} - Bid: {self.best_bid()}, Ask: {self.best_ask()}"


def parse_order_book(json_object: dict, symbol: str) -> OrderBook:
    """Parse JSON order book response into OrderBook object"""
    bids = []
    for level in json_object['bids']:
        bids.append(Tier(float(level[0]), float(level[1])))

    asks = []
    for level in json_object['asks']:
        asks.append(Tier(float(level[0]), float(level[1])))

    timestamp = float(json_object.get('T', time.time() * 1000)) / 1000
    return OrderBook(timestamp, bids, asks, symbol)


class OrderBookManager:
    def __init__(self, symbol_manager: BinanceSymbolManager):
        self.symbol_manager = symbol_manager
        self.order_books = {}  # Stores OrderBook objects by symbol

    def fetch_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """Fetch and parse order book for a single symbol"""
        response = requests.get(
            f"{BASE_URL}/depth",
            params={"symbol": symbol.upper(), "limit": limit}
        )

        if response.status_code == 200:
            order_book = parse_order_book(response.json(), symbol)
            self.order_books[symbol] = order_book
            return order_book
        else:
            logging.error(f"Error fetching {symbol}: {response.status_code} - {response.text}")
            return None

    def fetch_all_order_books(self, limit: int = 100):
        """Fetch order books for all tracked symbols"""
        for symbol in self.symbol_manager.get_symbols():
            self.fetch_order_book(symbol, limit)

    def print_top_of_book(self):
        """Print best bid/ask for all tracked symbols"""
        for symbol, ob in self.order_books.items():
            logging.info(f"{symbol}: Bid={ob.best_bid()}, Ask={ob.best_ask()}")


# Test
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Initialize managers
    symbol_manager = BinanceSymbolManager()
    order_book_manager = OrderBookManager(symbol_manager)

    # Add symbols to track
    symbol_manager.add_symbol("BTCUSDT")
    symbol_manager.add_symbol("ETHUSDT")
    symbol_manager.add_symbol("BNBUSDT")

    # Main loop
    while True:
        try:
            start_time = time.time()

            # Fetch all order books
            order_book_manager.fetch_all_order_books(limit=10)

            # Print current state
            order_book_manager.print_top_of_book()

            # Sleep to maintain ~1 second interval (account for request time)
            elapsed = time.time() - start_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

        except KeyboardInterrupt:
            logging.info("Stopping order book monitor...")
            break
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(5)  # Wait before retrying after error