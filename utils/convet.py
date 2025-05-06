import pandas as pd
from datetime import datetime

def convert_btc_price_data(csv_path, output_path=None):
    """
    Convert timestamp column in BTC price data to formatted date
    
    Args:
        csv_path: Path to the BTCPrice_H.csv file
        output_path: Optional path to save the converted CSV
        
    Returns:
        pandas.DataFrame: DataFrame with converted date column
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert timestamp (milliseconds) to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
     # Remove the timestamp column
    df = df.drop(columns=['timestamp'])
    
    # Reorder columns to put date first
    cols = ['date'] + [col for col in df.columns if col != 'date']
    df = df[cols]
    
    # Save to file if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Converted data saved to {output_path}")
    
    return df

# Example usage
# df = convert_btc_price_data('BTCPrice_H.csv', 'BTCPrice_H_with_date.csv')
# print(df.head())

df = convert_btc_price_data('./dataset/BTCPrice_H.csv', './BTCPrice_H_with_date.csv')
print(df.head())