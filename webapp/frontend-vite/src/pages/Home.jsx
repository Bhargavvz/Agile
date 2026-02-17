import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

const stats = [
    { value: "0.9876", label: "R² Score", icon: "📊" },
    { value: "6.71%", label: "MAPE", icon: "🎯" },
    { value: "340M", label: "Parameters", icon: "🧠" },
    { value: "< 10min", label: "Training Time", icon: "⚡" },
];

const features = [
    {
        icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M9 17H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-5" strokeLinecap="round" /><path d="M12 14l-3 7h6l-3-7z" strokeLinecap="round" strokeLinejoin="round" /></svg>,
        title: "BERT-Large NLP",
        desc: "Extracts deep semantic signals from Agile project reports, sprint retrospectives, and backlog descriptions.",
    },
    {
        icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" strokeLinecap="round" /></svg>,
        title: "Uncertainty Quantification",
        desc: "Monte Carlo Dropout decomposes prediction uncertainty into aleatoric and epistemic components.",
    },
    {
        icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 3v18h18" strokeLinecap="round" strokeLinejoin="round" /><path d="M7 16l4-8 4 4 4-6" strokeLinecap="round" strokeLinejoin="round" /></svg>,
        title: "Multi-Modal Fusion",
        desc: "Combines 1024-dim text embeddings with 18 structured features through late fusion for comprehensive analysis.",
    },
    {
        icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" strokeLinecap="round" strokeLinejoin="round" /></svg>,
        title: "Lightning Fast",
        desc: "Trained in under 10 minutes on NVIDIA H200. Real-time predictions with 90% confidence intervals.",
    },
    {
        icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" strokeLinecap="round" strokeLinejoin="round" /><path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" /></svg>,
        title: "Calibrated Confidence",
        desc: "Tells you not just the cost, but how much to trust the prediction — actionable for budget decisions.",
    },
    {
        icon: <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="2" y="3" width="20" height="14" rx="2" /><path d="M8 21h8" strokeLinecap="round" /><path d="M12 17v4" strokeLinecap="round" /></svg>,
        title: "8 Industry Domains",
        desc: "Validated across ecommerce, healthcare, fintech, edtech, logistics, social media, IoT, and gaming.",
    },
];

const fadeUp = {
    hidden: { opacity: 0, y: 30 },
    visible: (i = 0) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.6, ease: "easeOut" } }),
};

export default function Home() {
    return (
        <div className="min-h-screen flex flex-col">
            <Navbar />

            {/* ── Hero ── */}
            <section className="relative pt-48 pb-28 px-6 overflow-hidden">
                <div className="absolute top-28 left-10 w-80 h-80 bg-violet-600/20 rounded-full blur-[120px] animate-float" />
                <div className="absolute bottom-10 right-10 w-[28rem] h-[28rem] bg-indigo-600/15 rounded-full blur-[140px]" style={{ animationDelay: "3s" }} />
                <div className="absolute top-72 right-40 w-52 h-52 bg-cyan-500/10 rounded-full blur-[80px] animate-float" style={{ animationDelay: "1.5s" }} />

                <div className="max-w-5xl mx-auto text-center relative z-10">
                    <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={0}>
                        <span className="inline-flex items-center gap-2 px-5 py-2 rounded-full text-xs font-semibold bg-violet-500/15 text-violet-300 border border-violet-500/25 mb-10">
                            <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
                            Powered by BERT-Large · R² = 0.9876
                        </span>
                    </motion.div>

                    <motion.h1 initial="hidden" animate="visible" variants={fadeUp} custom={1} className="text-5xl md:text-7xl font-extrabold leading-[1.08] mb-8 tracking-tight">
                        Predict Software Costs{" "}
                        <span className="gradient-text">with Confidence</span>
                    </motion.h1>

                    <motion.p initial="hidden" animate="visible" variants={fadeUp} custom={2} className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-12 leading-relaxed">
                        A deep learning pipeline that reads your Agile project reports and structured metadata
                        to deliver dollar-valued cost forecasts with calibrated uncertainty bounds.
                    </motion.p>

                    <motion.div initial="hidden" animate="visible" variants={fadeUp} custom={3} className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <Link to="/predict" className="group px-8 py-4 rounded-2xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold text-lg shadow-2xl shadow-violet-500/25 hover:shadow-violet-500/40 hover:from-violet-500 hover:to-indigo-500 transition-all duration-300 flex items-center gap-3">
                            Start Estimating
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="group-hover:translate-x-1 transition-transform"><path d="M5 12h14" /><path d="M12 5l7 7-7 7" /></svg>
                        </Link>
                        <Link to="/about" className="px-8 py-4 rounded-2xl text-slate-300 font-medium text-lg border border-white/10 hover:bg-white/5 transition-all duration-300">
                            How It Works
                        </Link>
                    </motion.div>
                </div>
            </section>

            {/* ── Stats ── */}
            <section className="px-6 pb-24">
                <div className="max-w-4xl mx-auto">
                    <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.6 }} className="glass-strong p-8 rounded-2xl">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                            {stats.map((s, i) => (
                                <motion.div key={s.label} initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true }} transition={{ delay: i * 0.1, duration: 0.4 }} className="text-center">
                                    <div className="text-2xl mb-2">{s.icon}</div>
                                    <div className="text-2xl md:text-3xl font-extrabold gradient-text">{s.value}</div>
                                    <div className="text-xs text-slate-500 font-medium mt-1 uppercase tracking-wider">{s.label}</div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* ── Features ── */}
            <section className="px-6 pb-28">
                <div className="max-w-6xl mx-auto">
                    <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold mb-4">Why <span className="gradient-text">AgileScope AI</span>?</h2>
                        <p className="text-slate-400 max-w-2xl mx-auto">
                            Unlike traditional parametric models that ignore your project documentation,
                            our system reads and understands the narrative of your Agile workflow.
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {features.map((f, i) => (
                            <motion.div key={f.title} initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeUp} custom={i} className="glass-card p-8">
                                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-indigo-500/20 flex items-center justify-center text-violet-400 mb-5">{f.icon}</div>
                                <h3 className="text-lg font-semibold text-white mb-3">{f.title}</h3>
                                <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── CTA ── */}
            <section className="px-6 pb-28">
                <motion.div initial={{ opacity: 0, scale: 0.95 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true }} transition={{ duration: 0.5 }} className="max-w-4xl mx-auto glass-strong p-14 md:p-20 text-center rounded-3xl glow-purple">
                    <h2 className="text-3xl md:text-4xl font-bold mb-5">Ready to estimate your next project?</h2>
                    <p className="text-slate-400 mb-10 max-w-lg mx-auto">
                        Enter your project parameters or paste a report — get an instant cost prediction with uncertainty bounds.
                    </p>
                    <Link to="/predict" className="inline-flex items-center gap-2 px-8 py-4 rounded-2xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-semibold text-lg shadow-2xl shadow-violet-500/25 hover:shadow-violet-500/40 transition-all duration-300">
                        Launch Estimator
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14" /><path d="M12 5l7 7-7 7" /></svg>
                    </Link>
                </motion.div>
            </section>

            <Footer />
        </div>
    );
}
