# Agricultural Commodity Price Prediction System

## Overview
This project implements a robust predictive system for agricultural commodity prices in Telangana, India. It uses a hybrid approach combining **ARIMAX** (for linear trends and policy effects) and **LSTM** (for non-linear volatility and shocks).

## Project Structure
- `data/`: Contains raw and processed data.
- `src/etl/`: Scripts for data acquisition (AGMARKNET, Weather) and cleaning.
- `src/features/`: Feature engineering pipeline.
- `src/models/`: Implementation of Baseline (ARIMA), Deep Learning (LSTM), and Hybrid models.
- `config/`: Configuration files.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure API keys in `config/config.yaml` (optional for mock data).

## Usage

### 1. Data Acquisition
Fetch prices and weather data:
```bash
python src/etl/agmarknet.py
python src/etl/weather.py
python src/etl/static_data.py
```

### 2. Data Cleaning
Clean and merge datasets:
```bash
python src/etl/cleaning.py
```

### 3. Feature Engineering
Generate lagged and rolling features:
```bash
python src/features/build_features.py
```

### 4. Model Training & Evaluation
Train and compare models:
```bash
python src/models/evaluate.py
```

## Models Implemented
- **Baseline**: ARIMA(5,1,0)
- **Deep Learning**: LSTM with 2 layers
- **Hybrid**: ARIMAX-LSTM (Residual Modeling)

## Key Features
- **Climatic**: Cumulative rainfall, lagged temperature.
- **Economic**: MSP floor interaction, input costs.
- **Market**: Arrival momentum, daily volatility.
