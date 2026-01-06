import React from 'react';
import { Lightbulb, AlertOctagon, CheckCircle2 } from 'lucide-react';

const AdvisoryPanel = ({ riskLevel, modalPrice, priceMin }) => {
    const getAdvisory = () => {
        if (riskLevel === 'High') {
            return {
                title: "High Volatility Alert",
                message: "Market shows signs of instability. Prices may fluctuate significantly. Farmers are advised to hold produce if storage is available, or lock in prices via forward contracts if possible.",
                icon: <AlertOctagon className="w-6 h-6 text-rose-400" />,
                color: "border-rose-500/50 bg-rose-500/10"
            };
        } else if (riskLevel === 'Moderate') {
            return {
                title: "Cautious Market Outlook",
                message: "Moderate fluctuations expected. Monitor daily prices closely before bringing large quantities to market.",
                icon: <Lightbulb className="w-6 h-6 text-amber-400" />,
                color: "border-amber-500/50 bg-amber-500/10"
            };
        } else {
            return {
                title: "Stable Market Conditions",
                message: "Prices are expected to remain stable. Good time for regular trade activities.",
                icon: <CheckCircle2 className="w-6 h-6 text-emerald-400" />,
                color: "border-emerald-500/50 bg-emerald-500/10"
            };
        }
    };

    const advisory = getAdvisory();

    return (
        <div className={`p-6 rounded-2xl border backdrop-blur-sm ${advisory.color}`}>
            <div className="flex items-start gap-4">
                <div className="mt-1">{advisory.icon}</div>
                <div>
                    <h3 className="text-lg font-semibold text-white mb-2">{advisory.title}</h3>
                    <p className="text-slate-300 leading-relaxed text-sm">
                        {advisory.message}
                    </p>

                    {/* Dynamic MSP Check (Mock MSP for now) */}
                    <div className="mt-4 pt-4 border-t border-white/10">
                        <h4 className="text-xs font-bold text-white uppercase tracking-wider mb-2">Policy Insight</h4>
                        <p className="text-xs text-slate-400">
                            Predicted floor price (₹{priceMin?.toFixed(0)}) is
                            <span className="text-white font-medium ml-1">
                                {priceMin > 2000 ? "ABOVE" : "BELOW"}
                            </span> expected MSP benchmarks.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AdvisoryPanel;
