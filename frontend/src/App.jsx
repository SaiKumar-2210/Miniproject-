import React, { useState } from 'react';
import axios from 'axios';
import { CloudSun } from 'lucide-react';
import PredictionForm from './components/PredictionForm';
import RiskCard from './components/RiskCard';
import PriceChart from './components/PriceChart';
import AdvisoryPanel from './components/AdvisoryPanel';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartData, setChartData] = useState([]);

  const handlePredict = async ({ commodity, district }) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        commodity,
        district
      });
      setData(response.data);

      // Mock generate chart data based on result for demo
      // In prod, API should return history
      const mockHistory = [];
      const basePrice = response.data.modal_price;
      const volatility = response.data.volatility;

      // Generate 10 days of "history" leading up to forecast
      for (let i = 10; i > 0; i--) {
        mockHistory.push({
          date: `T-${i}`,
          price: basePrice * (1 + (Math.random() * volatility - volatility / 2)),
          forecast: null
        });
      }
      // Add T+1 Forecast
      mockHistory.push({
        date: 'Tomorrow',
        price: null,
        forecast: basePrice
      });

      setChartData(mockHistory);

    } catch (err) {
      setError(err.response?.data?.detail || "Failed to fetch prediction. Ensure backend is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 pb-20">
      {/* Header */}
      <div className="bg-slate-900 border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 py-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <CloudSun className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">AgriPrice <span className="text-indigo-400">Intelligence</span></h1>
              <p className="text-xs text-slate-400">Advanced Forecasting & Risk Assessment System</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

          {/* Left Sidebar: Controls */}
          <div className="lg:col-span-3 space-y-6">
            <PredictionForm onSubmit={handlePredict} isLoading={loading} />

            {/* Context/Info */}
            <div className="bg-gradient-to-br from-indigo-900/20 to-slate-900/50 p-6 rounded-2xl border border-indigo-500/20">
              <h3 className="text-indigo-400 font-semibold mb-2">Did you know?</h3>
              <p className="text-sm text-slate-400">
                Market volatility often spikes 2 weeks before harvest season. Check the Risk Gauge frequently.
              </p>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-9 space-y-6">

            {error && (
              <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-xl">
                Error: {error}
              </div>
            )}

            {!data && !loading && !error && (
              <div className="h-96 flex flex-col items-center justify-center text-slate-500 border-2 border-dashed border-slate-800 rounded-3xl">
                <CloudSun className="w-16 h-16 mb-4 opacity-20" />
                <p>Select a commodity and district to view forecasts</p>
              </div>
            )}

            {data && (
              <>
                <RiskCard
                  riskLevel={data.risk_level}
                  volatility={data.volatility}
                  modalPrice={data.modal_price}
                  priceMin={data.price_min}
                  priceMax={data.price_max}
                />

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <PriceChart data={chartData} commodity={data.commodity} district={data.district} />
                  <AdvisoryPanel
                    riskLevel={data.risk_level}
                    modalPrice={data.modal_price}
                    priceMin={data.price_min}
                  />
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
