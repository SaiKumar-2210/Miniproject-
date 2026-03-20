"""
Master Pipeline Script - Final Version
"""
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is on the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.utils.config_loader import load_config
from src.etl.agmarknet import AgmarknetClient
from src.etl.gemini_data import GeminiDataClient

print("="*60)
print("AGRICULTURAL COMMODITY PRICE PREDICTION PIPELINE")
print("="*60)

# Paths
raw_path = "data/raw"
processed_path = "data/processed"
os.makedirs(processed_path, exist_ok=True)

# Load project config
config = load_config()

# STEP 1: Load Data
print("\nSTEP 1: Loading Data...")
prices = pd.read_csv(os.path.join(raw_path, "agmarknet_data.csv"))
if 'arrival_date' in prices.columns:
    prices.rename(columns={'arrival_date': 'date'}, inplace=True)
weather = pd.read_csv(os.path.join(raw_path, "weather_data.csv"))
msp = pd.read_csv(os.path.join(raw_path, "msp_data.csv"))

print(f"  Prices (gov/mock): {prices.shape}")
print(f"  Weather: {weather.shape}")
print(f"  MSP: {msp.shape}")

# STEP 1b: Gemini Real-Time Data
print("\nSTEP 1b: Fetching Real-Time Data via Gemini API...")
try:
    gemini_client = GeminiDataClient(config)
    gemini_df = gemini_client.fetch_realtime_data(save=True)
    if not gemini_df.empty:
        prices = AgmarknetClient.merge_with_gemini(prices, gemini_df)
        print(f"  Prices after Gemini merge: {prices.shape}")
    else:
        print("  Gemini returned no data — using gov/mock prices only.")
except Exception as _gemini_err:
    print(f"  Gemini step failed (non-fatal): {_gemini_err}")

# STEP 2: Clean and Merge
print("\nSTEP 2: Cleaning and Merging...")

# Clean prices
prices['date'] = pd.to_datetime(prices['date'], format='mixed', dayfirst=True)

# Clean weather
weather['date'] = pd.to_datetime(weather['date'])
if weather['date'].dt.tz is not None:
    weather['date'] = weather['date'].dt.tz_localize(None)

# Merge
merged = pd.merge(prices, weather, on=['date', 'district'], how='left')
merged['year'] = merged['date'].dt.year

# Merge MSP
msp_long = pd.melt(msp, id_vars=['year'], var_name='commodity', value_name='msp_price')
final_df = pd.merge(merged, msp_long, on=['year', 'commodity'], how='left')

# Save merged
merged_path = os.path.join(processed_path, "merged_data.csv")
final_df.sort_values(by=['date', 'commodity', 'district'], ascending=[False, True, True], inplace=True)
final_df.to_csv(merged_path, index=False)
print(f"  Merged data: {final_df.shape}")

# STEP 3: Feature Engineering
print("\nSTEP 3: Feature Engineering...")

df = final_df.copy()
df.sort_values(by=['commodity', 'district', 'date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Time features
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Simple lagged features
df['price_lag_1'] = df['modal_price'].shift(1)
df['price_lag_7'] = df['modal_price'].shift(7)

# MSP ratio
df['msp_price_filled'] = df['msp_price'].fillna(df['modal_price'])
df['price_to_msp_ratio'] = df['modal_price'] / (df['msp_price_filled'] + 0.01)

# Drop rows with NaN in critical columns
df_clean = df.dropna(subset=['modal_price', 'commodity', 'district'])

# Save features
features_path = os.path.join(processed_path, "features_data.csv")
df_clean.to_csv(features_path, index=False)
print(f"  Features data: {df_clean.shape}")

# STEP 4: Summary
print("\n" + "="*60)
print("PIPELINE SUMMARY")
print("="*60)
print(f"Total records: {len(df_clean)}")
print(f"Commodities ({df_clean['commodity'].nunique()}): {', '.join(df_clean['commodity'].unique()[:5])}")
print(f"Districts ({df_clean['district'].nunique()}): {', '.join(df_clean['district'].unique())}")
print(f"Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")

print("\n" + "="*60)
print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nOutput files:")
print(f"  - {merged_path}")
print(f"  - {features_path}")
