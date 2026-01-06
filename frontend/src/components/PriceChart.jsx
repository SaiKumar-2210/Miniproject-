import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Area,
    ComposedChart
} from 'recharts';

const PriceChart = ({ data, commodity, district }) => {
    if (!data || data.length === 0) {
        return (
            <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 backdrop-blur-sm h-96 flex items-center justify-center">
                <p className="text-slate-500">No data available for visualization</p>
            </div>
        );
    }

    // Mock data generation for visualization if only single point provided
    // In real app, we'd fetch history + forecast series
    // Here we just visualize the single forecast point relative to a "mock" history or we need history from API
    // Since API only returns T+1, we will mock a small trend ending at T+1 for demo purposes

    // TODO: Update API to return history for plotting. For now, we mock.

    return (
        <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 backdrop-blur-sm">
            <h3 className="text-lg font-semibold text-white mb-6">Price Forecast Trend: {commodity} ({district})</h3>
            <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={data}>
                        <defs>
                            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" stroke="#64748b" tick={{ fill: '#64748b' }} />
                        <YAxis stroke="#64748b" tick={{ fill: '#64748b' }} domain={['auto', 'auto']} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }}
                            itemStyle={{ color: '#818cf8' }}
                        />
                        <Area type="monotone" dataKey="price" stroke="#6366f1" fillOpacity={1} fill="url(#colorPrice)" />
                        <Line type="monotone" dataKey="forecast" stroke="#f43f5e" strokeWidth={2} dot={{ r: 4 }} />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default PriceChart;
