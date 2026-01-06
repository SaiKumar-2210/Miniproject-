import pandas as pd
import numpy as np
import logging
import os
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config
from src.utils.model_registry import ModelRegistry
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GARCHModel:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['paths']['processed_data']
        self.output_path = os.path.join(config['paths']['models'], 'volatility')
        os.makedirs(self.output_path, exist_ok=True)
        self.registry = ModelRegistry()

    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(self.processed_path, "features_data.csv"))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except FileNotFoundError:
            logger.error("Features data not found.")
            return None

    def garch_likelihood(self, params, returns):
        """
        Negative Log-Likelihood for GARCH(1,1)
        params: [omega, alpha, beta]
        sigma^2_t = omega + alpha * returns_{t-1}^2 + beta * sigma^2_{t-1}
        """
        omega, alpha, beta = params
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns) # Initialize with sample variance
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
        # Log-likelihood
        # L = -0.5 * sum(log(sigma2) + returns^2 / sigma2)
        # We minimize negative log-likelihood
        log_lik = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
        return -log_lik

    def fit_garch(self, returns):
        """
        Fit GARCH(1,1) model.
        """
        # Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        bounds = ((1e-6, None), (1e-6, 1), (1e-6, 1))
        constraints = ({'type': 'ineq', 'fun': lambda x: 1 - x[1] - x[2]})
        
        # Initial guess
        initial_params = [np.var(returns)*0.01, 0.1, 0.8]
        
        result = minimize(self.garch_likelihood, initial_params, args=(returns,),
                          bounds=bounds, constraints=constraints)
        
        return result.x

    def predict_volatility(self, params, returns):
        """
        Predict conditional volatility.
        """
        omega, alpha, beta = params
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
        return np.sqrt(sigma2)

    def run_forecasting(self):
        df = self.load_data()
        if df is None: return
        
        commodities = self.config['commodities']
        districts = self.config['region']['districts']
        
        for commodity in commodities:
            for district in districts:
                subset = df[(df['commodity'] == commodity) & (df['district'] == district)].sort_values('date')
                if len(subset) < 100:
                    continue
                
                logger.info(f"Forecasting volatility for {commodity} in {district}...")
                
                # Calculate returns (log returns)
                subset['returns'] = np.log(subset['modal_price'] / subset['modal_price'].shift(1))
                returns = subset['returns'].dropna().values
                
                # Fit GARCH
                try:
                    params = self.fit_garch(returns)
                    omega, alpha, beta = params
                    logger.info(f"GARCH Parameters: Omega={omega:.6f}, Alpha={alpha:.4f}, Beta={beta:.4f}")
                    
                    # Predict
                    volatility = self.predict_volatility(params, returns)
                    
                    # Plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(subset['date'].iloc[1:], returns, alpha=0.5, label='Returns')
                    plt.plot(subset['date'].iloc[1:], volatility, color='red', label='Conditional Volatility (Sigma)')
                    plt.title(f'Volatility Forecast (GARCH) for {commodity}-{district}')
                    plt.legend()
                    plt.savefig(os.path.join(self.output_path, f'volatility_{commodity}_{district}.png'))
                    plt.close()
                    
                    # Save Params
                    params_dict = {
                        "omega": float(omega),
                        "alpha": float(alpha),
                        "beta": float(beta)
                    }
                    params_filename = f"{commodity}_{district}_garch.json"
                    params_path = os.path.join(self.output_path, params_filename)
                    with open(params_path, 'w') as f:
                        json.dump(params_dict, f, indent=4)
                        
                    logger.info(f"Saved GARCH params to {params_path}")
                    
                    # Register
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
                    rel_params_path = os.path.relpath(params_path, project_root)
                    
                    # We typically register this ALONGSIDE the main price model. 
                    # But since they are trained separately, we can use register_model to UPDATE the entry.
                    self.registry.register_model(
                        commodity=commodity,
                        district=district,
                        garch_params_path=rel_params_path
                    )
                    
                except Exception as e:
                    logger.error(f"GARCH failed for {commodity}-{district}: {e}")

if __name__ == "__main__":
    config = load_config()
    model = GARCHModel(config)
    model.run_forecasting()
