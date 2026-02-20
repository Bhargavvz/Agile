import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import './Home.css'

const fadeUp = {
    hidden: { opacity: 0, y: 30 },
    visible: (i) => ({
        opacity: 1, y: 0,
        transition: { delay: i * 0.15, duration: 0.6, ease: 'easeOut' }
    })
}

function AnimatedCounter({ end, suffix = '', prefix = '', duration = 2000 }) {
    const [count, setCount] = useState(0)
    useEffect(() => {
        let start = 0
        const step = end / (duration / 16)
        const timer = setInterval(() => {
            start += step
            if (start >= end) { setCount(end); clearInterval(timer) }
            else setCount(Math.floor(start * 100) / 100)
        }, 16)
        return () => clearInterval(timer)
    }, [end, duration])
    return <>{prefix}{typeof end === 'number' && end % 1 === 0 ? Math.floor(count).toLocaleString() : count.toFixed(2)}{suffix}</>
}

export default function Home() {
    const [metrics, setMetrics] = useState(null)

    useEffect(() => {
        fetch('/api/results/metrics').then(r => r.json()).then(setMetrics).catch(() => { })
    }, [])

    return (
        <div className="page home-page">
            <div className="container">
                {/* Hero */}
                <motion.section className="hero" initial="hidden" animate="visible">
                    <motion.div className="hero-badge" custom={0} variants={fadeUp}>
                        <span className="badge badge-cyan">🧠 Deep Learning Powered</span>
                    </motion.div>
                    <motion.h1 className="hero-title" custom={1} variants={fadeUp}>
                        Software Cost Estimation<br />
                        <span className="hero-gradient">in Agile Methodology</span>
                    </motion.h1>
                    <motion.p className="hero-subtitle" custom={2} variants={fadeUp}>
                        Predict project costs with confidence using BERT-Large NLP encoding,
                        multi-input regression, and uncertainty quantification.
                    </motion.p>
                    <motion.div className="hero-actions" custom={3} variants={fadeUp}>
                        <Link to="/predict" className="btn btn-primary">Try Prediction →</Link>
                        <Link to="/dashboard" className="btn btn-outline">View Dashboard</Link>
                    </motion.div>
                </motion.section>

                {/* Key Metrics */}
                {metrics && (
                    <motion.section className="key-metrics" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <div className="grid-4">
                            {[
                                { value: metrics.r2 * 100, suffix: '%', label: 'R² Score', icon: '📊' },
                                { value: metrics.mape_pct, suffix: '%', label: 'MAPE', icon: '🎯' },
                                { value: 10000, suffix: '', label: 'Training Samples', icon: '📁' },
                                { value: 340, suffix: 'M', label: 'BERT Parameters', icon: '🧠' },
                            ].map((s, i) => (
                                <motion.div className="stat-card" key={s.label} custom={i} variants={fadeUp}>
                                    <div className="stat-icon">{s.icon}</div>
                                    <div className="stat-value"><AnimatedCounter end={s.value} suffix={s.suffix} /></div>
                                    <div className="stat-label">{s.label}</div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.section>
                )}

                {/* Architecture Preview */}
                <motion.section className="arch-section" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                    <motion.h2 className="section-title" custom={0} variants={fadeUp}>
                        <span className="icon">🏗️</span> Model Architecture
                    </motion.h2>
                    <motion.div className="arch-card glass-card" custom={1} variants={fadeUp}>
                        <pre className="arch-diagram">{`
  ┌──────────────────────────────────────────────────────────────┐
  │  Input                                                  │
  │  ├── Report text  → BERT-Large → CLS token (1024-d)    │
  │  └── Numeric feats → MLP      → Embedding  (64-d)      │
  │                                                        │
  │  Fusion: Concatenate → (1088-dim)                      │
  │                                                        │
  │  Output                                                │
  │  ├── μ Head → Predicted Cost ($)                       │
  │  └── σ Head → Uncertainty (±$)                         │
  └────────────────────────────────────────────────────────┘`}
                        </pre>
                    </motion.div>
                </motion.section>

                {/* Feature Cards */}
                <motion.section className="features-section" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                    <motion.h2 className="section-title" custom={0} variants={fadeUp}>
                        <span className="icon">✨</span> Key Features
                    </motion.h2>
                    <div className="grid-3">
                        {[
                            { icon: '📝', title: 'NLP Report Analysis', desc: 'BERT-Large encodes natural-language Agile project reports into rich 1024-dim embeddings.' },
                            { icon: '🎯', title: 'Dual-Head Prediction', desc: 'Simultaneously predicts cost (μ) and uncertainty (σ) for every estimate.' },
                            { icon: '🔬', title: 'MC Dropout Inference', desc: '20 stochastic forward passes quantify epistemic uncertainty at inference time.' },
                            { icon: '📈', title: '2-Phase Training', desc: 'Phase 1 freezes BERT; Phase 2 fine-tunes end-to-end with discriminative learning rates.' },
                            { icon: '⚡', title: 'H200 GPU Optimised', desc: 'BFloat16 AMP, batch size 128, pinned memory — optimised for NVIDIA H200 GPUs.' },
                            { icon: '📊', title: '8 Evaluation Plots', desc: 'Publication-quality visualisations: loss curves, scatter plots, calibration, and more.' },
                        ].map((f, i) => (
                            <motion.div className="feature-card glass-card" key={f.title} custom={i + 1} variants={fadeUp}>
                                <div className="feature-icon">{f.icon}</div>
                                <h3>{f.title}</h3>
                                <p>{f.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </motion.section>
            </div>
        </div>
    )
}
