import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import PredictionForm from "../components/PredictionForm";
import ResultDisplay from "../components/ResultDisplay";

const API = "/api";

export default function Predict() {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [mode, setMode] = useState("form");

    const predict = async (data) => {
        setLoading(true); setError(null); setResult(null);
        try {
            const res = await fetch(`${API}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
            if (!res.ok) throw new Error(`Server error: ${res.status}`);
            setResult(await res.json());
        } catch (e) { setError(e.message); } finally { setLoading(false); }
    };

    const predictReport = async (text) => {
        setLoading(true); setError(null); setResult(null);
        try {
            const res = await fetch(`${API}/predict-report`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ report_text: text }) });
            if (!res.ok) throw new Error(`Server error: ${res.status}`);
            setResult(await res.json());
        } catch (e) { setError(e.message); } finally { setLoading(false); }
    };

    return (
        <div className="min-h-screen flex flex-col">
            <Navbar />
            <main className="flex-1 pt-40 pb-16 px-6">
                <div className="max-w-6xl mx-auto">
                    {/* Header */}
                    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="text-center mb-10">
                        <h1 className="text-4xl md:text-5xl font-extrabold mb-3">Cost <span className="gradient-text">Estimator</span></h1>
                        <p className="text-slate-400 max-w-xl mx-auto">Enter your project details or paste a report to get an AI-powered cost prediction with uncertainty analysis.</p>
                    </motion.div>

                    {/* Mode toggle */}
                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.4 }} className="flex items-center justify-center gap-2 mb-10">
                        <button onClick={() => { setMode("form"); setResult(null); }}
                            className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${mode === "form" ? "bg-violet-600/20 text-violet-300 border border-violet-500/30" : "text-slate-500 hover:text-slate-300 border border-transparent"}`}>
                            📋 Structured Input
                        </button>
                        <button onClick={() => { setMode("report"); setResult(null); }}
                            className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${mode === "report" ? "bg-violet-600/20 text-violet-300 border border-violet-500/30" : "text-slate-500 hover:text-slate-300 border border-transparent"}`}>
                            📄 Paste Report
                        </button>
                    </motion.div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Input */}
                        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3, duration: 0.5 }}>
                            {mode === "form" ? <PredictionForm onSubmit={predict} loading={loading} /> : <ReportInput onSubmit={predictReport} loading={loading} />}
                        </motion.div>

                        {/* Results */}
                        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4, duration: 0.5 }}>
                            <AnimatePresence mode="wait">
                                {loading && (
                                    <motion.div key="load" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                                        className="glass-strong p-12 rounded-2xl flex flex-col items-center justify-center min-h-[400px]">
                                        <div className="w-16 h-16 rounded-full border-4 border-violet-500/20 border-t-violet-500 animate-spin mb-6" />
                                        <p className="text-slate-400 text-lg font-medium">Running MC Dropout inference...</p>
                                        <p className="text-slate-600 text-sm mt-2">20 stochastic forward passes</p>
                                    </motion.div>
                                )}
                                {error && !loading && (
                                    <motion.div key="err" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                                        className="glass-strong p-8 rounded-2xl border border-red-500/20">
                                        <div className="flex items-center gap-3 mb-3"><span className="text-2xl">⚠️</span><h3 className="text-lg font-semibold text-red-400">Prediction Failed</h3></div>
                                        <p className="text-sm text-slate-400">{error}</p>
                                        <p className="text-xs text-slate-600 mt-3">Make sure the backend is running: <code className="text-violet-400">python app.py</code></p>
                                    </motion.div>
                                )}
                                {result && !loading && (
                                    <motion.div key="res" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.4 }}>
                                        <ResultDisplay result={result} />
                                    </motion.div>
                                )}
                                {!result && !loading && !error && (
                                    <motion.div key="ph" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                                        className="glass-strong p-12 rounded-2xl flex flex-col items-center justify-center min-h-[400px] text-center">
                                        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-violet-500/10 to-indigo-500/10 flex items-center justify-center mb-6">
                                            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="opacity-60">
                                                <line x1="12" y1="1" x2="12" y2="23" /><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                                            </svg>
                                        </div>
                                        <h3 className="text-xl font-semibold text-slate-300 mb-2">Your Prediction</h3>
                                        <p className="text-sm text-slate-500 max-w-xs">Fill in the project details and click predict to see the AI-generated cost estimation with uncertainty.</p>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.div>
                    </div>
                </div>
            </main>
            <Footer />
        </div>
    );
}

function ReportInput({ onSubmit, loading }) {
    const [text, setText] = useState("");
    const sample = `Project Overview\n================\nDomain: Healthcare\nComplexity Level: High\nTeam Size: 12\nDuration: 18 months\n\nSprint Summary\n==============\nTotal Sprints: 36 sprints\nSprint Length: 2 weeks\nTotal User Stories: backlog contains 85 user stories\nAverage story points: 8.5\nVelocity: velocity of 32 points per sprint\n\nTechnology & Risk\n=================\nTech Stack: Advanced (ML pipelines, real-time data processing, HIPAA encryption)\nRequirement Volatility: High volatility due to evolving regulatory requirements\nRisk Level: High risk — integrating with legacy hospital EHR systems`;

    return (
        <div className="glass-strong p-8 rounded-2xl">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-white">📄 Paste Report</h2>
                <button onClick={() => setText(sample)} className="px-3 py-1.5 rounded-lg text-xs font-medium text-violet-400 bg-violet-500/10 hover:bg-violet-500/20 transition-colors">Load Sample</button>
            </div>
            <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste your Agile project report here..."
                className="w-full h-80 bg-white/3 border border-white/8 rounded-xl p-4 text-sm text-slate-300 placeholder-slate-600 resize-none focus:outline-none focus:border-violet-500/40 transition-colors font-mono" />
            <button onClick={() => text.trim() && onSubmit(text)} disabled={loading || !text.trim()}
                className="mt-5 w-full py-4 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:from-violet-500 hover:to-indigo-500 transition-all duration-300 shadow-lg hover:shadow-violet-500/25 flex items-center justify-center gap-2 text-base">
                {loading ? <><div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />Analyzing Report...</> : <>🧠 Analyze & Predict</>}
            </button>
        </div>
    );
}
