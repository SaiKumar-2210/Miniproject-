import pandas as pd
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StaticDataLoader:
    def __init__(self, config):
        self.config = config

    def load_msp_data(self):
        """
        Load Minimum Support Price (MSP) data.
        In a real scenario, this might come from a CSV or API.
        Here we'll create a small reference dataset for the target commodities.
        """
        # Example MSP data (approximate values for recent years)
        data = {
            "year": [2020, 2021, 2022, 2023, 2024],
            "Rice": [1868, 1940, 2040, 2183, 2300], # Common Grade
            "Maize": [1850, 1870, 1962, 2090, 2225],
            "Cotton": [5515, 5726, 6080, 6620, 7121], # Medium Staple
            "Red Gram": [6000, 6300, 6600, 7000, 7550]
        }
        df = pd.DataFrame(data)
        return df

    def load_production_estimates(self):
        """
        Load production estimates for Telangana.
        """
        # Example production data in Lakh Tonnes
        data = {
            "year": [2020, 2021, 2022, 2023],
            "Rice": [100.5, 120.2, 160.1, 174.4],
            "Maize": [15.2, 18.5, 22.1, 25.0],
            "Cotton": [45.0, 48.2, 50.5, 52.0], # Lakh Bales usually, simplified here
            "Red Gram": [2.5, 2.8, 3.1, 3.5]
        }
        df = pd.DataFrame(data)
        return df

    def save_static_data(self):
        msp_df = self.load_msp_data()
        prod_df = self.load_production_estimates()
        
        raw_path = self.config['paths']['raw_data']
        os.makedirs(raw_path, exist_ok=True)
        msp_df.to_csv(os.path.join(raw_path, "msp_data.csv"), index=False)
        prod_df.to_csv(os.path.join(raw_path, "production_data.csv"), index=False)
        logger.info("Static data saved.")

if __name__ == "__main__":
    config = load_config()
    loader = StaticDataLoader(config)
    loader.save_static_data()
