import json
import pandas as pd
import os

def main():
    # Read the trades from the JSON file
    file_path = 'trades.json'
    
    if not os.path.exists(file_path):
        print("No trades file found.")
        return
    
    with open(file_path, 'r') as f:
        trades_data = json.load(f)
    
    trades = trades_data.get('trades', [])
    
    print(f"Number of trades: {len(trades)}")
    
    # Analyze trade directions
    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']
    
    print(f"\nLong trades: {len(long_trades)}")
    print(f"Short trades: {len(short_trades)}")
    
    if len(trades) > 0:
        print(f"\nLong trades percentage: {len(long_trades)/len(trades)*100:.1f}%")
        print(f"Short trades percentage: {len(short_trades)/len(trades)*100:.1f}%")
    
    # Analyze performance by direction
    if len(long_trades) > 0:
        long_profits = [t['profit'] for t in long_trades]
        print(f"\nLong trades:")
        print(f"  Total profit: ${sum(long_profits):.2f}")
        print(f"  Average profit: ${sum(long_profits)/len(long_profits):.2f}")
        print(f"  Win rate: {len([p for p in long_profits if p > 0])/len(long_profits)*100:.1f}%")
    
    if len(short_trades) > 0:
        short_profits = [t['profit'] for t in short_trades]
        print(f"\nShort trades:")
        print(f"  Total profit: ${sum(short_profits):.2f}")
        print(f"  Average profit: ${sum(short_profits)/len(short_trades):.2f}")
        print(f"  Win rate: {len([p for p in short_profits if p > 0])/len(short_profits)*100:.1f}%")
    
    # Calculate overall performance
    total_profit = sum(t['profit'] for t in trades)
    average_trade = total_profit / len(trades) if len(trades) > 0 else 0
    win_rate = len([t for t in trades if t['profit'] > 0]) / len(trades) * 100 if len(trades) > 0 else 0
    
    print(f"\nOverall performance:")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Average trade: ${average_trade:.2f}")
    print(f"Win rate: {win_rate:.1f}%")

if __name__ == "__main__":
    main()
