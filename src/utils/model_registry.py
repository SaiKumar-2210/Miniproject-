import json
import os
import logging
import pandas as pd
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry.json"):
        # If path is relative, make it relative to project root
        if not os.path.isabs(registry_path):
             # Assuming this file is in src/utils/, project root is ../../
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.registry_path = os.path.join(base_dir, registry_path)
        else:
            self.registry_path = registry_path
            
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.registry = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupt registry file at {self.registry_path}. Starting fresh.")
                self.registry = {}
        else:
            self.registry = {}

    def save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def register_model(self, commodity: str, district: str, 
                       model_type: Optional[str] = None, 
                       model_path: Optional[str] = None, 
                       scaler_path: Optional[str] = None, 
                       garch_params_path: Optional[str] = None, **kwargs):
        """
        Register a model for a specific commodity and district.
        """
        key = f"{commodity}_{district}"
        
        # Start with existing entry if available, else empty
        entry = self.registry.get(key, {}).copy()
        
        # Update fields if provided
        if model_type:
            entry["model_type"] = model_type
        if model_path:
            entry["model_path"] = model_path
            
        entry["updated_at"] = str(pd.Timestamp.now())
        
        if scaler_path:
            entry["scaler_path"] = scaler_path
            
        if garch_params_path:
            entry["garch_params_path"] = garch_params_path
            
        # Update with any additional kwargs
        entry.update(kwargs)
        
        self.registry[key] = entry
            
        self.save_registry()
        logger.info(f"Registered model for {key}")

    def get_model_entry(self, commodity: str, district: str) -> Optional[Dict]:
        key = f"{commodity}_{district}"
        return self.registry.get(key)

if __name__ == "__main__":
    # Test
    import pandas as pd # Needed for timestamp in test
    registry = ModelRegistry()
    print(f"Registry loaded from: {registry.registry_path}")
