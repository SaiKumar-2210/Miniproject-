import React, { useState } from 'react';
import { Search } from 'lucide-react';

const commodities = ["Rice", "Maize", "Cotton", "Red Gram"];
const districts = ["Warangal", "Karimnagar", "Nalgonda", "Khammam", "Nizamabad"];

const PredictionForm = ({ onSubmit, isLoading }) => {
    const [commodity, setCommodity] = useState(commodities[0]);
    const [district, setDistrict] = useState(districts[0]);

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit({ commodity, district });
    };

    return (
        <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 backdrop-blur-sm">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                <Search className="w-5 h-5 text-indigo-400" />
                Prediction Parameters
            </h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label className="block text-sm font-medium text-slate-400 mb-1">Commodity</label>
                    <select
                        value={commodity}
                        onChange={(e) => setCommodity(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-700 rounded-lg py-2.5 px-3 text-white focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all outline-none appearance-none"
                    >
                        {commodities.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-slate-400 mb-1">District</label>
                    <select
                        value={district}
                        onChange={(e) => setDistrict(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-700 rounded-lg py-2.5 px-3 text-white focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all outline-none appearance-none"
                    >
                        {districts.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                </div>

                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 rounded-lg shadow-lg shadow-indigo-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed mt-4"
                >
                    {isLoading ? 'Generating Forecast...' : 'Generate Forecast'}
                </button>
            </form>
        </div>
    );
};

export default PredictionForm;
