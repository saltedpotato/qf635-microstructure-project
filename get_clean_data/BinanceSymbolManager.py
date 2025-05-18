class BinanceSymbolManager:
    def __init__(self):
        """Initialize an empty set to store symbols (avoids duplicates)."""
        self.symbols = set()

    def add_symbol(self, symbol: str) -> str:
        """Add a symbol to the tracker (e.g., 'BTCUSDT')."""
        symbol = symbol.upper().strip()  # Normalize input
        if symbol in self.symbols:
            return f"'{symbol}' already exists in the list."
        self.symbols.add(symbol)
        return f"'{symbol}' added successfully."

    def remove_symbol(self, symbol: str) -> str:
        """Remove a symbol from the tracker."""
        symbol = symbol.upper().strip()
        if symbol not in self.symbols:
            return f"'{symbol}' not found in the list."
        self.symbols.remove(symbol)
        return f"'{symbol}' removed successfully."

    def get_symbols(self) -> list:
        """Return all tracked symbols as a sorted list."""
        return sorted(self.symbols)

    def has_symbol(self, symbol: str) -> bool:
        """Check if a symbol exists in the tracker."""
        return symbol.upper().strip() in self.symbols

    def clear_symbols(self) -> str:
        """Clear all symbols from the tracker."""
        self.symbols.clear()
        return "All symbols cleared."

    def __str__(self) -> str:
        """String representation of the tracked symbols."""
        return f"Tracked Symbols: {sorted(self.symbols)}"

# if __name__ == "__main__":
#     manager = BinanceSymbolManager()
#
#     # Add symbols
#     print(manager.add_symbol("BTCUSDT"))  # 'BTCUSDT' added successfully.
#     print(manager.add_symbol("ETHUSDT"))  # 'ETHUSDT' added successfully.
#     print(manager.add_symbol("BTCUSDT"))  # 'BTCUSDT' already exists.
#
#     # Remove a symbol
#     print(manager.remove_symbol("ETHUSDT"))  # 'ETHUSDT' removed successfully.
#     print(manager.remove_symbol("DOGEUSDT"))  # 'DOGEUSDT' not found.
#
#     # Get all symbols
#     print(manager.get_symbols())  # ['BTCUSDT']
#
#     # Check if a symbol exists
#     print(manager.has_symbol("btcusdt"))  # True (case-insensitive)
#     print(manager.has_symbol("XRPUSDT"))  # False
#
#     # Clear all symbols
#     print(manager.clear_symbols())  # All symbols cleared.
#     print(manager.get_symbols())  # []