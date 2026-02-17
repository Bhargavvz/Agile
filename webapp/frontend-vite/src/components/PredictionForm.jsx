import { useState } from "react";

const defaults = {
    team_size: 8, duration_months: 6, num_sprints: 12,
    total_user_stories: 24, avg_story_points: 5, velocity_per_sprint: 25,
    complexity_level: "Medium", tech_stack_difficulty: "Intermediate",
    requirement_volatility: "Medium", risk_level: "Medium", report_text: "",
};

const sliders = [
    { key: "team_size", label: "Team Size", min: 1, max: 50, step: 1, unit: " devs", icon: "👥" },
    { key: "duration_months", label: "Duration", min: 1, max: 48, step: 1, unit: " months", icon: "📅" },
    { key: "num_sprints", label: "Sprints", min: 1, max: 100, step: 1, unit: " sprints", icon: "🔄" },
    { key: "total_user_stories", label: "User Stories", min: 1, max: 500, step: 1, unit: " stories", icon: "📝" },
    { key: "avg_story_points", label: "Avg. Story Points", min: 1, max: 21, step: 0.5, unit: " pts", icon: "⭐" },
    { key: "velocity_per_sprint", label: "Sprint Velocity", min: 1, max: 100, step: 1, unit: " pts/sprint", icon: "🚀" },
];

const selects = [
    { key: "complexity_level", label: "Complexity", opts: ["Low", "Medium", "High"] },
    { key: "tech_stack_difficulty", label: "Tech Stack", opts: ["Basic", "Intermediate", "Advanced"] },
    { key: "requirement_volatility", label: "Req. Volatility", opts: ["Low", "Medium", "High"] },
    { key: "risk_level", label: "Risk Level", opts: ["Low", "Medium", "High"] },
];

export default function PredictionForm({ onSubmit, loading }) {
    const [v, setV] = useState(defaults);
    const set = (k, val) => setV((p) => ({ ...p, [k]: val }));

    const submit = (e) => {
        e.preventDefault();
        onSubmit({
            ...v,
            team_size: Number(v.team_size), duration_months: Number(v.duration_months),
            num_sprints: Number(v.num_sprints), total_user_stories: Number(v.total_user_stories),
            avg_story_points: Number(v.avg_story_points), velocity_per_sprint: Number(v.velocity_per_sprint),
        });
    };

    return (
        <form onSubmit={submit} className="glass-strong p-8 rounded-2xl">
            <h2 className="text-lg font-semibold text-white mb-7 flex items-center gap-2">📋 Project Parameters</h2>

            {/* Sliders */}
            <div className="space-y-6 mb-8">
                {sliders.map(({ key, label, min, max, step, unit, icon }) => (
                    <div key={key}>
                        <div className="flex items-center justify-between mb-2">
                            <label className="text-sm font-medium text-slate-400 flex items-center gap-2"><span>{icon}</span> {label}</label>
                            <span className="text-sm font-bold text-white tabular-nums">{v[key]}{unit}</span>
                        </div>
                        <input type="range" min={min} max={max} step={step} value={v[key]} onChange={(e) => set(key, parseFloat(e.target.value))} className="w-full" />
                        <div className="flex justify-between mt-1"><span className="text-[10px] text-slate-600">{min}</span><span className="text-[10px] text-slate-600">{max}</span></div>
                    </div>
                ))}
            </div>

            {/* Dropdowns */}
            <div className="grid grid-cols-2 gap-4 mb-8">
                {selects.map(({ key, label, opts }) => (
                    <div key={key}>
                        <label className="text-sm font-medium text-slate-400 block mb-2">{label}</label>
                        <select value={v[key]} onChange={(e) => set(key, e.target.value)}
                            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-violet-500/40 transition-colors cursor-pointer">
                            {opts.map((o) => <option key={o} value={o} className="bg-slate-900">{o}</option>)}
                        </select>
                    </div>
                ))}
            </div>

            {/* Report */}
            <div className="mb-7">
                <label className="text-sm font-medium text-slate-400 block mb-2">📄 Report Text <span className="text-slate-600">(optional — improves accuracy)</span></label>
                <textarea value={v.report_text} onChange={(e) => set("report_text", e.target.value)}
                    placeholder="Paste sprint retrospective or project report for NLP-enhanced prediction..."
                    className="w-full h-28 bg-white/3 border border-white/8 rounded-xl p-3 text-xs text-slate-300 placeholder-slate-600 resize-none focus:outline-none focus:border-violet-500/40 transition-colors font-mono" />
            </div>

            <button type="submit" disabled={loading}
                className="w-full py-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:from-violet-500 hover:to-indigo-500 transition-all duration-300 shadow-lg hover:shadow-violet-500/25 flex items-center justify-center gap-2 text-base">
                {loading ? <><div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />Predicting...</> : <>⚡ Predict Cost</>}
            </button>
        </form>
    );
}
