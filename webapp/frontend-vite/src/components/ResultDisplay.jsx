import { motion } from "framer-motion";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { useState, useEffect } from "react";

function AnimNum({ value, prefix = "", suffix = "" }) {
    const [d, setD] = useState(0);
    useEffect(() => {
        const start = performance.now();
        const tick = (now) => {
            const p = Math.min((now - start) / 1500, 1);
            setD(value * (1 - Math.pow(1 - p, 3)));
            if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }, [value]);
    return <span className="tabular-nums">{prefix}{d.toLocaleString("en-US", { maximumFractionDigits: 0 })}{suffix}</span>;
}

function Gauge({ percent }) {
    const r = 54, c = 2 * Math.PI * r, off = c - (percent / 100) * c;
    const col = percent >= 90 ? "#10b981" : percent >= 70 ? "#f59e0b" : "#ef4444";
    return (
        <div className="relative w-36 h-36 mx-auto">
            <svg width="144" height="144" viewBox="0 0 144 144" className="-rotate-90">
                <circle cx="72" cy="72" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="9" />
                <motion.circle cx="72" cy="72" r={r} fill="none" stroke={col} strokeWidth="9" strokeLinecap="round" strokeDasharray={c} initial={{ strokeDashoffset: c }} animate={{ strokeDashoffset: off }} transition={{ duration: 1.5, ease: "easeOut" }} />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-bold" style={{ color: col }}>{Math.round(percent)}%</span>
                <span className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">Confidence</span>
            </div>
        </div>
    );
}

const COLORS = ["#7c3aed", "#06b6d4"];

export default function ResultDisplay({ result }) {
    if (!result) return null;

    const uncData = [
        { name: "Aleatoric", value: result.aleatoric_uncertainty_usd },
        { name: "Epistemic", value: result.epistemic_uncertainty_usd },
    ];

    const uncPct = result.predicted_cost_usd > 0 ? ((result.total_uncertainty_usd / result.predicted_cost_usd) * 100).toFixed(1) : 0;

    return (
        <div className="space-y-5">
            {/* Cost card */}
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass-strong p-10 rounded-2xl text-center glow-purple">
                <p className="text-sm text-slate-500 uppercase tracking-wider mb-3 font-medium">Estimated Cost</p>
                <div className="text-4xl md:text-5xl font-extrabold gradient-text mb-3"><AnimNum value={result.predicted_cost_usd} prefix="$" /></div>
                <p className="text-sm text-slate-500">± <AnimNum value={result.total_uncertainty_usd} prefix="$" /> ({uncPct}%)</p>
            </motion.div>

            {/* Confidence + Donut */}
            <div className="grid grid-cols-2 gap-5">
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card p-6 rounded-2xl">
                    <Gauge percent={result.confidence_percent} />
                </motion.div>

                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass-card p-6 rounded-2xl">
                    <p className="text-xs text-slate-500 text-center mb-3 font-medium uppercase tracking-wider">Uncertainty Split</p>
                    <div className="h-28">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie data={uncData} cx="50%" cy="50%" innerRadius={28} outerRadius={44} paddingAngle={4} dataKey="value" animationDuration={1200}>
                                    {uncData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                                </Pie>
                                <Tooltip contentStyle={{ background: "rgba(15,12,41,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px", fontSize: "12px", color: "#f1f5f9" }} formatter={(v) => `$${v.toLocaleString()}`} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="flex justify-center gap-4 mt-2">
                        <span className="flex items-center gap-1.5 text-[11px] text-slate-500"><span className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS[0] }} /> Aleatoric</span>
                        <span className="flex items-center gap-1.5 text-[11px] text-slate-500"><span className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS[1] }} /> Epistemic</span>
                    </div>
                </motion.div>
            </div>

            {/* CI band */}
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="glass-card p-7 rounded-2xl">
                <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-5">90% Confidence Interval</p>
                <div className="relative h-10">
                    <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-2 rounded-full bg-white/5" />
                    {(() => {
                        const range = result.ci_90_upper_usd - result.ci_90_lower_usd;
                        const total = result.ci_90_upper_usd * 1.1;
                        const leftPct = (result.ci_90_lower_usd / total) * 100;
                        const widthPct = (range / total) * 100;
                        const midPct = ((result.predicted_cost_usd - result.ci_90_lower_usd) / range) * widthPct + leftPct;
                        return (<>
                            <motion.div initial={{ width: 0 }} animate={{ width: `${widthPct}%` }} transition={{ delay: 0.6, duration: 0.8, ease: "easeOut" }} className="absolute top-1/2 -translate-y-1/2 h-4 rounded-full bg-gradient-to-r from-violet-500/30 to-indigo-500/30 border border-violet-500/20" style={{ left: `${leftPct}%` }} />
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }} className="absolute top-1/2 w-4 h-4 rounded-full bg-violet-400 shadow-lg shadow-violet-500/50" style={{ left: `${midPct}%`, transform: "translate(-50%, -50%)" }} />
                        </>);
                    })()}
                </div>
                <div className="flex justify-between mt-4">
                    <div><span className="text-xs text-slate-600 block">Low</span><span className="text-sm font-semibold text-emerald-400">${result.ci_90_lower_usd.toLocaleString("en-US", { maximumFractionDigits: 0 })}</span></div>
                    <div className="text-center"><span className="text-xs text-slate-600 block">Predicted</span><span className="text-sm font-bold text-violet-400">${result.predicted_cost_usd.toLocaleString("en-US", { maximumFractionDigits: 0 })}</span></div>
                    <div className="text-right"><span className="text-xs text-slate-600 block">High</span><span className="text-sm font-semibold text-amber-400">${result.ci_90_upper_usd.toLocaleString("en-US", { maximumFractionDigits: 0 })}</span></div>
                </div>
            </motion.div>

            {/* Detail cards */}
            <div className="grid grid-cols-3 gap-4">
                {[
                    { label: "Aleatoric", val: result.aleatoric_uncertainty_usd, color: "text-violet-400", sub: "Data noise" },
                    { label: "Epistemic", val: result.epistemic_uncertainty_usd, color: "text-cyan-400", sub: "Model gap" },
                    { label: "Total", val: result.total_uncertainty_usd, color: "text-amber-400", sub: "Combined" },
                ].map((c, i) => (
                    <motion.div key={c.label} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 + i * 0.05 }} className="glass-card p-5 rounded-xl text-center">
                        <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">{c.label}</p>
                        <p className={`text-lg font-bold ${c.color}`}>${c.val.toLocaleString("en-US", { maximumFractionDigits: 0 })}</p>
                        <p className="text-[10px] text-slate-600 mt-1">{c.sub}</p>
                    </motion.div>
                ))}
            </div>
        </div>
    );
}
