import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
    ResponsiveContainer, AreaChart, Area
} from 'recharts'
import './Dashboard.css'

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.5 } })
}

export default function Dashboard() {
    const [metrics, setMetrics] = useState(null)
    const [history, setHistory] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        Promise.all([
            fetch('/api/results/metrics').then(r => r.json()),
            fetch('/api/results/history').then(r => r.json()),
        ]).then(([m, h]) => {
            setMetrics(m)
            setHistory(h)
            setLoading(false)
        }).catch(() => setLoading(false))
    }, [])

    if (loading) return <div className="page"><div className="container"><div className="spinner" /></div></div>

    // Prepare chart data
    const chartData = history ? history.train_loss.map((_, i) => ({
        epoch: i + 1,
        trainLoss: parseFloat(history.train_loss[i].toFixed(4)),
        valLoss: parseFloat(history.val_loss[i].toFixed(4)),
        valR2: parseFloat((history.val_r2[i] * 100).toFixed(2)),
        valMAPE: parseFloat(history.val_mape[i].toFixed(2)),
        epochTime: parseFloat(history.epoch_time[i].toFixed(1)),
    })) : []

    const phase1 = chartData.slice(0, 5)
    const phase2 = chartData.slice(5)

    return (
        <div className="page dashboard-page">
            <div className="container">
                <motion.div className="page-header" initial="hidden" animate="visible">
                    <motion.h1 custom={0} variants={fadeUp}>Training Dashboard</motion.h1>
                    <motion.p custom={1} variants={fadeUp}>
                        25-epoch training performance with 2-phase strategy
                    </motion.p>
                </motion.div>

                {/* Metric Cards */}
                {metrics && (
                    <motion.div className="grid-4 metric-cards" initial="hidden" animate="visible">
                        {[
                            { label: 'R² Score', value: (metrics.r2 * 100).toFixed(2) + '%', color: 'cyan' },
                            { label: 'MAPE', value: metrics.mape_pct.toFixed(2) + '%', color: 'emerald' },
                            { label: 'MAE (USD)', value: '$' + metrics.mae_usd.toLocaleString(undefined, { maximumFractionDigits: 0 }), color: 'amber' },
                            { label: 'RMSE (USD)', value: '$' + metrics.rmse_usd.toLocaleString(undefined, { maximumFractionDigits: 0 }), color: 'violet' },
                        ].map((m, i) => (
                            <motion.div className={`stat-card border-${m.color}`} key={m.label} custom={i} variants={fadeUp}>
                                <div className="stat-value">{m.value}</div>
                                <div className="stat-label">{m.label}</div>
                            </motion.div>
                        ))}
                    </motion.div>
                )}

                {/* Training Curves */}
                <motion.div className="charts-grid" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                    {/* Loss Curves */}
                    <motion.div className="chart-card glass-card" custom={0} variants={fadeUp}>
                        <h3 className="chart-title">📉 Loss Curves</h3>
                        <p className="chart-subtitle">Gaussian NLL loss — Phase 1 (frozen) → Phase 2 (fine-tune)</p>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                                <YAxis stroke="#64748b" fontSize={12} />
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9' }}
                                />
                                <Legend />
                                <Line type="monotone" dataKey="trainLoss" stroke="#06d6f2" strokeWidth={2} dot={false} name="Train Loss" />
                                <Line type="monotone" dataKey="valLoss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Val Loss" />
                            </LineChart>
                        </ResponsiveContainer>
                    </motion.div>

                    {/* R² Curve */}
                    <motion.div className="chart-card glass-card" custom={1} variants={fadeUp}>
                        <h3 className="chart-title">📈 Validation R²</h3>
                        <p className="chart-subtitle">R² score progression over 25 epochs</p>
                        <ResponsiveContainer width="100%" height={300}>
                            <AreaChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                                <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} unit="%" />
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9' }}
                                    formatter={(v) => v + '%'}
                                />
                                <Area type="monotone" dataKey="valR2" stroke="#10b981" fill="rgba(16,185,129,0.15)" strokeWidth={2} name="Val R²" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </motion.div>

                    {/* MAPE Curve */}
                    <motion.div className="chart-card glass-card" custom={2} variants={fadeUp}>
                        <h3 className="chart-title">🎯 Validation MAPE</h3>
                        <p className="chart-subtitle">Mean Absolute Percentage Error</p>
                        <ResponsiveContainer width="100%" height={300}>
                            <AreaChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                                <YAxis stroke="#64748b" fontSize={12} unit="%" />
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9' }}
                                    formatter={(v) => v + '%'}
                                />
                                <Area type="monotone" dataKey="valMAPE" stroke="#f43f5e" fill="rgba(244,63,94,0.15)" strokeWidth={2} name="Val MAPE" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </motion.div>

                    {/* Epoch Time */}
                    <motion.div className="chart-card glass-card" custom={3} variants={fadeUp}>
                        <h3 className="chart-title">⏱️ Epoch Duration</h3>
                        <p className="chart-subtitle">Phase 1 (~10s) vs Phase 2 (~26s) — BERT unfreezing visible</p>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                                <YAxis stroke="#64748b" fontSize={12} unit="s" />
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9' }}
                                    formatter={(v) => v + 's'}
                                />
                                <Line type="stepAfter" dataKey="epochTime" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Time (s)" />
                            </LineChart>
                        </ResponsiveContainer>
                    </motion.div>
                </motion.div>

                {/* Training Strategy */}
                <motion.div className="strategy-section" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                    <motion.h2 className="section-title" custom={0} variants={fadeUp}>
                        <span className="icon">🎓</span> 2-Phase Training Strategy
                    </motion.h2>
                    <div className="grid-2">
                        <motion.div className="glass-card strategy-card" custom={1} variants={fadeUp}>
                            <div className="strategy-phase">
                                <span className="badge badge-cyan">Phase 1</span>
                                <span className="strategy-epochs">Epochs 1–5</span>
                            </div>
                            <h3>❄️ BERT Frozen</h3>
                            <p>Train only the regression heads while BERT-Large weights remain frozen. Establishes a strong regression baseline.</p>
                            <ul>
                                <li>Learning rate: 2×10⁻⁵ (head only)</li>
                                <li>~10 seconds per epoch</li>
                                <li>R² reaches ~79% by epoch 5</li>
                            </ul>
                        </motion.div>
                        <motion.div className="glass-card strategy-card" custom={2} variants={fadeUp}>
                            <div className="strategy-phase">
                                <span className="badge badge-rose">Phase 2</span>
                                <span className="strategy-epochs">Epochs 6–25</span>
                            </div>
                            <h3>🔥 Full Fine-Tuning</h3>
                            <p>Unfreeze BERT-Large and fine-tune entire model with discriminative learning rates.</p>
                            <ul>
                                <li>BERT LR: 5×10⁻⁶ · Head LR: 2×10⁻⁵</li>
                                <li>~26 seconds per epoch</li>
                                <li>R² reaches 98.76% by epoch 25</li>
                            </ul>
                        </motion.div>
                    </div>
                </motion.div>
            </div>
        </div>
    )
}
