import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgmarknetClient:
    def __init__(self, config):
        self.config = config
        self.api_key = config['api_keys'].get('gov_data')
        self.base_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070" # Example Resource ID
        self.state = config['region']['state']
        self.commodities = config['commodities']

    def fetch_data(self, date_from, date_to=None, limit=1000, use_mock=False):
        """
        Fetch data from OGD API.
        Note: This is a placeholder structure. The actual OGD API requires specific parameters.
        If no API key is present, it returns mock data.
        """
        if use_mock or not self.api_key:
            if use_mock:
                logger.info("Mock data generation requested.")
            else:
                logger.warning("No API key found. Generating mock data.")
            return self._generate_mock_data(date_from, date_to)

        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "filters[state]": self.state,
            # "filters[commodity]": commodity # Loop through commodities if needed
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            records = data.get('records', [])
            df = pd.DataFrame(records)
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def _generate_mock_data(self, date_from, date_to):
        """
        Generate synthetic data for testing pipeline.
        """
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")
        
        dates = pd.date_range(start=date_from, end=date_to)
        data = []
        
        import numpy as np
        
        for date in dates:
            for commodity in self.commodities:
                for district in self.config['region']['districts']:
                    base_price = 2000 if commodity == 'Rice' else 5000
                    volatility = 100
                    
                    modal_price = base_price + np.random.normal(0, volatility)
                    min_price = modal_price - np.random.uniform(50, 150)
                    max_price = modal_price + np.random.uniform(50, 150)
                    
                    record = {
                        "date": date.strftime("%Y-%m-%d"),
                        "state": self.state,
                        "district": district,
                        "market": f"{district} Mandi",
                        "commodity": commodity,
                        "min_price": round(min_price, 2),
                        "max_price": round(max_price, 2),
                        "modal_price": round(modal_price, 2),
                        "arrival_quantity": round(np.random.uniform(10, 100), 2)
                    }
                    data.append(record)
                    
        return pd.DataFrame(data)

    def save_data(self, df, filename="agmarknet_data.csv"):
        path = os.path.join(self.config['paths']['raw_data'], filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Data saved to {path}")

if __name__ == "__main__":
    config = load_config()
    client = AgmarknetClient(config)
    
    # Fetch last 30 days of data
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    df = client.fetch_data(date_from=start_date)
    
    if not df.empty:
        print(df.head())
        client.save_data(df)
    else:
        print("No data fetched.")
