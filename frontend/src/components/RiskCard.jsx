import React from 'react';
import { AlertTriangle, TrendingUp, TrendingDown, ShieldCheck } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const RiskCard = ({ riskLevel, volatility, modalPrice, priceMin, priceMax }) => {
    const getRiskColor = (level) => {
        switch (level?.toLowerCase()) {
            case 'low': return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20';
            case 'moderate': return 'text-amber-400 bg-amber-400/10 border-amber-400/20';
            case 'high': return 'text-rose-400 bg-rose-400/10 border-rose-400/20';
            default: return 'text-slate-400 bg-slate-400/10 border-slate-400/20';
        }
    };

    const riskStyle = getRiskColor(riskLevel);

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Modal Price Card */}
            <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 backdrop-blur-sm">
                <h3 className="text-slate-400 text-sm font-medium mb-1">Forecasted Modal Price</h3>
                <div className="text-3xl font-bold text-white">
                    ₹{modalPrice?.toLocaleString('en-IN') || '---'}
                    <span className="text-xs font-normal text-slate-500 ml-2">/ Quintal</span>
                </div>
            </div>

            {/* Risk Level Card */}
            <div className={twMerge("p-6 rounded-2xl border backdrop-blur-sm flex flex-col justify-center", riskStyle)}>
                <div className="flex items-center gap-3">
                    <AlertTriangle className="w-8 h-8" />
                    <div>
                        <h3 className="text-sm font-medium opacity-80">Market Risk</h3>
                        <div className="text-2xl font-bold uppercase tracking-wide">{riskLevel || 'Unknown'}</div>
                    </div>
                </div>
            </div>

            {/* Volatility Card */}
            <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 backdrop-blur-sm">
                <h3 className="text-slate-400 text-sm font-medium mb-1">Volatility Index (Sigma)</h3>
                <div className="text-2xl font-bold text-indigo-400 flex items-center gap-2">
                    {(volatility * 100).toFixed(2)}%
                    <TrendingUp className="w-5 h-5 opacity-50" />
                </div>
                <p className="text-xs text-slate-500 mt-2">Expected price fluctuation range</p>
            </div>

            {/* Price Range Card */}
            <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 backdrop-blur-sm">
                <h3 className="text-slate-400 text-sm font-medium mb-1">Expected Price Band</h3>
                <div className="space-y-1">
                    <div className="flex justify-between items-center text-sm">
                        <span className="text-slate-500">Ceiling</span>
                        <span className="text-emerald-400 font-semibold">₹{priceMax?.toFixed(0)}</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div className="bg-gradient-to-r from-rose-500 via-amber-500 to-emerald-500 h-full w-full opacity-50"></div>
                    </div>
                    <div className="flex justify-between items-center text-sm">
                        <span className="text-slate-500">Floor</span>
                        <span className="text-rose-400 font-semibold">₹{priceMin?.toFixed(0)}</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default RiskCard;
