import pandas as pd

def main():
    # Read the CSV file
    df = pd.read_csv('reports/institutional/institutional_trades.csv')
    
    print("=== Trade Analysis ===")
    print(f"Total trades: {len(df)}")
    print(f"Long trades: {len(df[df['direction'] == 'long'])}")
    print(f"Short trades: {len(df[df['direction'] == 'short'])}")
    
    # Calculate profits
    long_profit = df[df['direction'] == 'long']['profit'].sum()
    short_profit = df[df['direction'] == 'short']['profit'].sum()
    
    print(f"\nLong trades profit: ${long_profit:.2f}")
    print(f"Short trades profit: ${short_profit:.2f}")
    print(f"Total profit: ${long_profit + short_profit:.2f}")
    
    # Calculate win rates
    long_wins = len(df[(df['direction'] == 'long') & (df['profit'] > 0)])
    short_wins = len(df[(df['direction'] == 'short') & (df['profit'] > 0)])
    
    if len(df[df['direction'] == 'long']) > 0:
        print(f"\nLong win rate: {long_wins / len(df[df['direction'] == 'long']) * 100:.1f}%")
    
    if len(df[df['direction'] == 'short']) > 0:
        print(f"Short win rate: {short_wins / len(df[df['direction'] == 'short']) * 100:.1f}%")
    
    # Average trade profit
    avg_long = long_profit / len(df[df['direction'] == 'long']) if len(df[df['direction'] == 'long']) > 0 else 0
    avg_short = short_profit / len(df[df['direction'] == 'short']) if len(df[df['direction'] == 'short']) > 0 else 0
    
    print(f"\nAverage long trade: ${avg_long:.2f}")
    print(f"Average short trade: ${avg_short:.2f}")

if __name__ == "__main__":
    main()
