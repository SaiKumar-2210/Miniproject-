"""Debug Pipeline"""
import pandas as pd
import numpy as np
import os

raw_path = "data/raw"
processed_path = "data/processed"
os.makedirs(processed_path, exist_ok=True)

print("Loading...")
prices = pd.read_csv(os.path.join(raw_path, "agmarknet_data.csv"))
weather = pd.read_csv(os.path.join(raw_path, "weather_data.csv"))
msp = pd.read_csv(os.path.join(raw_path, "msp_data.csv"))

print(f"Prices: {prices.shape}")
print(f"Weather: {weather.shape}")
print(f"MSP: {msp.shape}")

# Clean
if 'arrival_date' in prices.columns:
    prices.rename(columns={'arrival_date': 'date'}, inplace=True)

print("\nConverting dates...")
prices['date'] = pd.to_datetime(prices['date'])
weather['date'] = pd.to_datetime(weather['date'])

print("\nMerging prices and weather...")
try:
    merged = pd.merge(prices, weather, on=['date', 'district'], how='left')
    print(f"After merge 1: {merged.shape}")
except Exception as e:
    print(f"Merge 1 failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nAdding year...")
merged['year'] = merged['date'].dt.year

print("\nMelting MSP...")
try:
    msp_long = pd.melt(msp, id_vars=['year'], var_name='commodity', value_name='msp_price')
    print(f"MSP melted: {msp_long.shape}")
except Exception as e:
    print(f"Melt failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nMerging with MSP...")
try:
    final_df = pd.merge(merged, msp_long, on=['year', 'commodity'], how='left')
    print(f"Final: {final_df.shape}")
except Exception as e:
    print(f"Merge 2 failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nSaving...")
merged_path = os.path.join(processed_path, "merged_data.csv")
final_df.to_csv(merged_path, index=False)

print(f"\n✓ Success! Saved to {merged_path}")
print(f"Shape: {final_df.shape}")
print(f"\nColumns: {final_df.columns.tolist()}")
