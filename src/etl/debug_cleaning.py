import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config

config = load_config()
raw_path = config['paths']['raw_data']

print("Loading prices...")
prices = pd.read_csv(os.path.join(raw_path, "agmarknet_data.csv"))
print("Prices shape:", prices.shape)
print("Columns:", prices.columns)
print(prices.head())

if 'arrival_date' in prices.columns:
    prices.rename(columns={'arrival_date': 'date'}, inplace=True)
prices['date'] = pd.to_datetime(prices['date'])
print("Date converted.")
print("Dtypes:\n", prices.dtypes)

print("Sorting...")
prices = prices.sort_values(by=['commodity', 'district', 'date'])

print("FFill...")
try:
    prices['modal_price'] = prices.groupby(['commodity', 'district'])['modal_price'].ffill()
    print("FFill success.")
except Exception as e:
    print(f"FFill failed: {e}")

print("Grouping...")
cleaned_dfs = []
for name, group in prices.groupby('commodity'):
    print(f"Group: {name}, shape: {group.shape}")
    Q1 = group['modal_price'].quantile(0.25)
    Q3 = group['modal_price'].quantile(0.75)
    print(f"Q1: {Q1}, Q3: {Q3}")
    cleaned_dfs.append(group)

print(f"Cleaned DFS count: {len(cleaned_dfs)}")
if cleaned_dfs:
    res = pd.concat(cleaned_dfs)
    print("Concat success. Shape:", res.shape)
else:
    print("Cleaned DFS empty.")
