import { motion } from 'framer-motion'
import './About.css'

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.5 } })
}

export default function About() {
    return (
        <div className="page about-page">
            <div className="container">
                <motion.div className="page-header" initial="hidden" animate="visible">
                    <motion.h1 custom={0} variants={fadeUp}>About This Project</motion.h1>
                    <motion.p custom={1} variants={fadeUp}>
                        Software Cost Estimation in Agile Methodology using Deep Learning
                    </motion.p>
                </motion.div>

                <div className="about-content">
                    {/* Overview */}
                    <motion.section className="about-section glass-card" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <motion.h2 custom={0} variants={fadeUp}>🎯 Overview</motion.h2>
                        <motion.p custom={1} variants={fadeUp}>
                            This project trains a neural network to read natural-language Agile project reports and predict
                            the development cost in USD — along with a confidence interval quantifying model uncertainty.
                            It combines BERT-Large NLP encoding with structured feature processing through a multi-input
                            regression architecture with dual-head output for both cost prediction and uncertainty estimation.
                        </motion.p>
                    </motion.section>

                    {/* Architecture */}
                    <motion.section className="about-section glass-card" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <motion.h2 custom={0} variants={fadeUp}>🏗️ Model Architecture</motion.h2>
                        <motion.div custom={1} variants={fadeUp}>
                            <pre className="arch-text">{`Report text  → BERT-Large (340M params) → CLS token → 1024-dim
Numeric feats → Feature MLP (128 → 64)  →             64-dim
                                                        ↓
                         Concatenation (1088-dim)
                            ↓              ↓
                    μ Head (cost)    σ Head (uncertainty)
                   1088→512→256→1    1088→256→1 (softplus)`}</pre>
                        </motion.div>
                    </motion.section>

                    {/* Methodology */}
                    <motion.section className="about-section glass-card" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <motion.h2 custom={0} variants={fadeUp}>🔬 Methodology</motion.h2>
                        <div className="method-grid">
                            {[
                                {
                                    title: 'Dataset Generation',
                                    desc: '10,000 synthetic Agile project reports with deterministic cost labels. Cost = f(team_size, duration, stories, complexity, tech_stack, volatility).',
                                },
                                {
                                    title: '2-Phase Training',
                                    desc: 'Phase 1 (5 epochs): Freeze BERT, train regression head. Phase 2 (20 epochs): Unfreeze BERT, fine-tune end-to-end with discriminative learning rates.',
                                },
                                {
                                    title: 'Uncertainty Quantification',
                                    desc: 'Aleatoric uncertainty via σ head (Gaussian NLL loss). Epistemic uncertainty via MC Dropout (20 forward passes). Total = √(aleatoric² + epistemic²).',
                                },
                                {
                                    title: 'Evaluation',
                                    desc: 'R² = 98.76%, MAPE = 6.71%, MAE = $62K, RMSE = $120K. 8 publication-quality plots including calibration and prediction intervals.',
                                },
                            ].map((m, i) => (
                                <motion.div key={m.title} className="method-item" custom={i + 1} variants={fadeUp}>
                                    <h3>{m.title}</h3>
                                    <p>{m.desc}</p>
                                </motion.div>
                            ))}
                        </div>
                    </motion.section>

                    {/* Tech Stack */}
                    <motion.section className="about-section glass-card" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <motion.h2 custom={0} variants={fadeUp}>🛠️ Technology Stack</motion.h2>
                        <div className="tech-grid">
                            {[
                                { category: 'Deep Learning', items: ['PyTorch', 'Transformers (HuggingFace)', 'BERT-Large (340M)'] },
                                { category: 'Data & Analysis', items: ['pandas', 'NumPy', 'scikit-learn'] },
                                { category: 'Visualization', items: ['matplotlib', 'seaborn', 'Recharts'] },
                                { category: 'Web Stack', items: ['FastAPI', 'React + Vite', 'Framer Motion'] },
                            ].map((t, i) => (
                                <motion.div key={t.category} className="tech-item" custom={i + 1} variants={fadeUp}>
                                    <h3>{t.category}</h3>
                                    <ul>
                                        {t.items.map(item => <li key={item}>{item}</li>)}
                                    </ul>
                                </motion.div>
                            ))}
                        </div>
                    </motion.section>

                    {/* Cost Formula */}
                    <motion.section className="about-section glass-card" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <motion.h2 custom={0} variants={fadeUp}>📊 Cost Label Generation</motion.h2>
                        <motion.div custom={1} variants={fadeUp}>
                            <pre className="formula-text">{`final_effort = (team_size × duration × 160 + stories × avg_sp × 4)
             × complexity_mult × tech_mult × volatility_mult

final_cost = final_effort × cost_per_hour × (1 ± 5% noise)

Complexity:  Low=1.0   Medium=1.35  High=1.65
Tech Stack:  Basic=1.0 Intermediate=1.25 Advanced=1.5
Volatility:  Low=1.0   Medium=1.10  High=1.25`}</pre>
                        </motion.div>
                    </motion.section>
                </div>
            </div>
        </div>
    )
}
