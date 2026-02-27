import pandas as pd
import os

# Check if the file exists
csv_file = 'reports/institutional/institutional_trades.csv'
if not os.path.exists(csv_file):
    print("File not found:", csv_file)
else:
    try:
        trades = pd.read_csv(csv_file)
        print("Number of trades:", len(trades))
        print("\nTrades:")
        for i, row in trades.iterrows():
            print(f"Trade {i+1}:")
            print(f"  Timestamp: {row['timestamp']}")
            print(f"  Direction: {row['direction']}")
            print(f"  Position Size: {row['position_size']:.2f}")
            print(f"  Confidence Score: {row['confidence_score']:.2f}")
            print(f"  Profit: {row['profit']:.2f}")
            print(f"  Exit Reason: {row['exit_reason']}")
            print()
    except Exception as e:
        print("Error reading file:", str(e))
