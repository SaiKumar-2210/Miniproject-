import pandas as pd
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config
from src.models.baselines import BaselineModel
from src.models.deep_learning import DeepLearningModel
from src.models.hybrid import HybridModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.results = []

    def run_evaluation(self):
        commodities = self.config['commodities']
        districts = self.config['region']['districts']
        
        # Initialize models
        baseline = BaselineModel(self.config)
        dl_model = DeepLearningModel(self.config)
        hybrid_model = HybridModel(self.config)
        
        for commodity in commodities:
            for district in districts:
                logger.info(f"Evaluating for {commodity} in {district}...")
                
                # Baseline
                try:
                    b_rmse, b_mape = baseline.train_evaluate(commodity, district)
                except Exception as e:
                    logger.error(f"Baseline failed: {e}")
                    b_rmse, b_mape = None, None
                
                # Deep Learning
                try:
                    d_rmse, d_mape = dl_model.train_evaluate(commodity, district)
                except Exception as e:
                    logger.error(f"DL failed: {e}")
                    d_rmse, d_mape = None, None
                
                # Hybrid
                try:
                    h_rmse, h_mape = hybrid_model.train_evaluate(commodity, district)
                except Exception as e:
                    logger.error(f"Hybrid failed: {e}")
                    h_rmse, h_mape = None, None
                
                b_accuracy = max(0.0, 100.0 - (b_mape * 100.0)) if b_mape is not None else None
                d_accuracy = max(0.0, 100.0 - (d_mape * 100.0)) if d_mape is not None else None
                h_accuracy = max(0.0, 100.0 - (h_mape * 100.0)) if h_mape is not None else None
                
                self.results.append({
                    "Commodity": commodity,
                    "District": district,
                    "Baseline_RMSE": b_rmse, "Baseline_Accuracy(%)": b_accuracy,
                    "GRU_RMSE": d_rmse, "GRU_Accuracy(%)": d_accuracy,
                    "Hybrid_RMSE": h_rmse, "Hybrid_Accuracy(%)": h_accuracy
                })
        
        # Save results
        results_df = pd.DataFrame(self.results)
        output_path = os.path.join(self.config['paths']['models'], "evaluation_results.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        # Print Summary
        print("\n=== Evaluation Summary ===")
        print(results_df.groupby('Commodity')[['Baseline_Accuracy(%)', 'GRU_Accuracy(%)', 'Hybrid_Accuracy(%)']].mean())
        
        return results_df

if __name__ == "__main__":
    config = load_config()
    evaluator = Evaluator(config)
    evaluator.run_evaluation()
