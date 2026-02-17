import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i = 0) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" } }),
};

const arch = [
    { label: "Agile Report (Text)", color: "from-orange-500/20 to-amber-500/20", border: "border-orange-500/20", desc: "Sprint retrospectives, backlog descriptions, standup notes" },
    { label: "WordPiece Tokenizer", color: "from-blue-500/20 to-indigo-500/20", border: "border-blue-500/20", desc: "Subword tokenization with max 512 tokens" },
    { label: "BERT-Large Encoder", color: "from-blue-600/30 to-indigo-600/30", border: "border-blue-500/30", desc: "340M parameters → 1024-dim CLS embedding" },
    { label: "Feature MLP (18 → 64)", color: "from-teal-500/20 to-cyan-500/20", border: "border-teal-500/20", desc: "Structured features: team size, sprints, velocity, etc." },
    { label: "Fusion Layer (1088-dim)", color: "from-violet-500/20 to-purple-500/20", border: "border-violet-500/20", desc: "Text + feature embeddings concatenated" },
    { label: "MC Dropout (p=0.1)", color: "from-red-500/10 to-pink-500/10", border: "border-red-500/20", desc: "20 stochastic passes for Bayesian uncertainty" },
    { label: "μ Head → Cost ($)", color: "from-emerald-500/20 to-green-500/20", border: "border-emerald-500/20", desc: "1088 → 512 → 256 → 1" },
    { label: "σ Head → Uncertainty", color: "from-amber-500/20 to-yellow-500/20", border: "border-amber-500/20", desc: "1088 → 256 → 1 with softplus" },
];

const metrics = [
    { label: "R² Score", value: "0.9876", desc: "98.76% variance explained" },
    { label: "MAPE", value: "6.71%", desc: "Avg. within 7% of true cost" },
    { label: "MAE", value: "$62,075", desc: "Mean Absolute Error" },
    { label: "RMSE", value: "$119,945", desc: "Root Mean Squared Error" },
    { label: "Median AE", value: "$21,275", desc: "Half within $21K of true cost" },
    { label: "Training", value: "9.5 min", desc: "25 epochs on NVIDIA H200" },
];

const baselines = [
    { method: "Linear Regression", r2: 0.7821, mape: 31.45 },
    { method: "Random Forest", r2: 0.8934, mape: 18.72 },
    { method: "Gradient Boosting", r2: 0.9102, mape: 15.38 },
    { method: "SVR (RBF)", r2: 0.8567, mape: 22.14 },
    { method: "MLP (3-layer)", r2: 0.8789, mape: 19.56 },
    { method: "Ours (BERT + Features)", r2: 0.9876, mape: 6.71, hl: true },
];

export default function About() {
    return (
        <div className="min-h-screen flex flex-col">
            <Navbar />
            <main className="flex-1 pt-40 pb-16 px-6">
                <div className="max-w-5xl mx-auto">

                    {/* Header */}
                    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="text-center mb-16">
                        <h1 className="text-4xl md:text-5xl font-extrabold mb-5">About the <span className="gradient-text">Model</span></h1>
                        <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">
                            A dual-input deep learning architecture that fuses BERT-Large text understanding
                            with structured project metadata to predict Agile software costs with calibrated uncertainty.
                        </p>
                    </motion.div>

                    {/* Architecture */}
                    <motion.section initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} className="mb-20">
                        <h2 className="text-2xl font-bold mb-7 flex items-center gap-3">
                            <span className="w-9 h-9 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-sm">🏗️</span> Architecture
                        </h2>
                        <div className="space-y-3">
                            {arch.map((b, i) => (
                                <motion.div key={b.label} initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={i}
                                    className={`glass-card p-5 rounded-xl flex items-center gap-5 border ${b.border}`}>
                                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${b.color} flex items-center justify-center shrink-0`}>
                                        <span className="text-xs font-bold text-white/70">{i + 1}</span>
                                    </div>
                                    <div className="flex-1">
                                        <h3 className="font-semibold text-white text-sm">{b.label}</h3>
                                        <p className="text-xs text-slate-500 mt-0.5">{b.desc}</p>
                                    </div>
                                    {i < arch.length - 1 && (
                                        <div className="text-slate-700"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M19 12l-7 7-7-7" /></svg></div>
                                    )}
                                </motion.div>
                            ))}
                        </div>
                    </motion.section>

                    {/* Metrics */}
                    <motion.section initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} className="mb-20">
                        <h2 className="text-2xl font-bold mb-7 flex items-center gap-3">
                            <span className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center text-sm">📊</span> Performance Metrics
                        </h2>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-5">
                            {metrics.map((m, i) => (
                                <motion.div key={m.label} initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={i}
                                    className="glass-card p-6 rounded-xl">
                                    <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">{m.label}</p>
                                    <p className="text-2xl font-extrabold gradient-text mt-1">{m.value}</p>
                                    <p className="text-[11px] text-slate-600 mt-2 leading-relaxed">{m.desc}</p>
                                </motion.div>
                            ))}
                        </div>
                    </motion.section>

                    {/* Comparison */}
                    <motion.section initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} className="mb-20">
                        <h2 className="text-2xl font-bold mb-7 flex items-center gap-3">
                            <span className="w-9 h-9 rounded-lg bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-sm">⚔️</span> Baseline Comparison
                        </h2>
                        <div className="glass-strong rounded-2xl overflow-hidden">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-white/5">
                                        <th className="text-left px-6 py-4 text-xs text-slate-500 uppercase tracking-wider font-medium">Method</th>
                                        <th className="text-right px-6 py-4 text-xs text-slate-500 uppercase tracking-wider font-medium">R²</th>
                                        <th className="text-right px-6 py-4 text-xs text-slate-500 uppercase tracking-wider font-medium">MAPE (%)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {baselines.map((c) => (
                                        <tr key={c.method} className={`border-b border-white/3 ${c.hl ? "bg-violet-500/5" : ""}`}>
                                            <td className={`px-6 py-3.5 ${c.hl ? "font-bold text-violet-300" : "text-slate-400"}`}>{c.hl && "🏆 "}{c.method}</td>
                                            <td className={`text-right px-6 py-3.5 tabular-nums ${c.hl ? "font-bold text-violet-300" : "text-slate-400"}`}>{c.r2.toFixed(4)}</td>
                                            <td className={`text-right px-6 py-3.5 tabular-nums ${c.hl ? "font-bold text-violet-300" : "text-slate-400"}`}>{c.mape.toFixed(2)}%</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </motion.section>

                    {/* Uncertainty */}
                    <motion.section initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} className="mb-20">
                        <h2 className="text-2xl font-bold mb-7 flex items-center gap-3">
                            <span className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-sm">🎯</span> Uncertainty Explained
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                            <div className="glass-card p-7 rounded-xl border-l-4 border-l-violet-500">
                                <h3 className="font-semibold text-violet-300 mb-3">Aleatoric Uncertainty</h3>
                                <p className="text-sm text-slate-400 leading-relaxed">Captures <strong>inherent data noise</strong> — irreducible variability. Large values mean projects of this type are naturally variable.</p>
                            </div>
                            <div className="glass-card p-7 rounded-xl border-l-4 border-l-cyan-500">
                                <h3 className="font-semibold text-cyan-300 mb-3">Epistemic Uncertainty</h3>
                                <p className="text-sm text-slate-400 leading-relaxed">Captures <strong>model knowledge gaps</strong> — uncertainty that more training data could reduce. Signals data collection needs.</p>
                            </div>
                        </div>
                    </motion.section>

                    {/* Tech stack */}
                    <motion.section initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp}>
                        <h2 className="text-2xl font-bold mb-7 flex items-center gap-3">
                            <span className="w-9 h-9 rounded-lg bg-gradient-to-br from-pink-500 to-rose-600 flex items-center justify-center text-sm">🛠️</span> Technology Stack
                        </h2>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            {[
                                { n: "PyTorch", d: "Model training & inference" }, { n: "BERT-Large", d: "340M param NLP encoder" },
                                { n: "FastAPI", d: "Python REST API server" }, { n: "React + Vite", d: "Lightning-fast frontend" },
                                { n: "Tailwind CSS", d: "Utility-first styling" }, { n: "Recharts", d: "React chart library" },
                                { n: "Framer Motion", d: "Animation framework" }, { n: "NVIDIA H200", d: "141 GB HBM3e GPU" },
                            ].map((t, i) => (
                                <motion.div key={t.n} initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={i}
                                    className="glass-card p-5 rounded-xl text-center">
                                    <p className="font-semibold text-white text-sm">{t.n}</p>
                                    <p className="text-[10px] text-slate-600 mt-1">{t.d}</p>
                                </motion.div>
                            ))}
                        </div>
                    </motion.section>

                </div>
            </main>
            <Footer />
        </div>
    );
}
