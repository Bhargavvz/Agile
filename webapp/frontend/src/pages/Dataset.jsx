import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'
import './Dataset.css'

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.5 } })
}

const COLORS = ['#06d6f2', '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#f43f5e', '#ec4899', '#14b8a6']

export default function Dataset() {
    const [stats, setStats] = useState(null)
    const [importance, setImportance] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        Promise.all([
            fetch('/api/dataset/stats').then(r => r.json()),
            fetch('/api/results/feature-importance').then(r => r.json()),
        ]).then(([s, fi]) => {
            setStats(s)
            setImportance(fi)
            setLoading(false)
        }).catch(() => setLoading(false))
    }, [])

    if (loading) return <div className="page"><div className="container"><div className="spinner" /></div></div>

    // Prepare chart data
    const domainData = stats ? Object.entries(stats.domains).map(([name, count]) => ({ name, count })) : []
    const complexityData = stats ? Object.entries(stats.complexity_levels).map(([name, count]) => ({ name, count })) : []
    const avgCostData = stats ? Object.entries(stats.domain_avg_cost).map(([name, cost]) => ({ name, cost: Math.round(cost) })) : []
    const importanceData = importance ? Object.entries(importance).map(([name, value]) => ({
        name: name.replace(/_/g, ' ').replace(/=/g, ': '),
        value: parseFloat(value.toFixed(4))
    })) : []

    return (
        <div className="page dataset-page">
            <div className="container">
                <motion.div className="page-header" initial="hidden" animate="visible">
                    <motion.h1 custom={0} variants={fadeUp}>Dataset Explorer</motion.h1>
                    <motion.p custom={1} variants={fadeUp}>
                        10,000 synthetic Agile Scrum project reports across 8 domains
                    </motion.p>
                </motion.div>

                {/* Overview Stats */}
                {stats && (
                    <motion.div className="grid-4 dataset-stats" initial="hidden" animate="visible">
                        {[
                            { icon: '📁', value: stats.total_projects.toLocaleString(), label: 'Total Projects' },
                            { icon: '🏢', value: Object.keys(stats.domains).length, label: 'Domains' },
                            { icon: '💰', value: '$' + (stats.cost.mean / 1000).toFixed(0) + 'K', label: 'Mean Cost' },
                            { icon: '⏱️', value: stats.duration_months.mean + ' mo', label: 'Mean Duration' },
                        ].map((s, i) => (
                            <motion.div className="stat-card" key={s.label} custom={i} variants={fadeUp}>
                                <div className="stat-icon">{s.icon}</div>
                                <div className="stat-value">{s.value}</div>
                                <div className="stat-label">{s.label}</div>
                            </motion.div>
                        ))}
                    </motion.div>
                )}

                <div className="charts-2col">
                    {/* Domain Distribution */}
                    <motion.div className="chart-card glass-card" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <motion.h3 className="chart-title" custom={0} variants={fadeUp}>🏢 Domain Distribution</motion.h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie data={domainData} dataKey="count" nameKey="name" cx="50%" cy="50%"
                                    outerRadius={100} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                    labelLine={true} stroke="none"
                                >
                                    {domainData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Pie>
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9' }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </motion.div>

                    {/* Average Cost by Domain */}
                    <motion.div className="chart-card glass-card" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <motion.h3 className="chart-title" custom={0} variants={fadeUp}>💰 Average Cost by Domain</motion.h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={avgCostData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis type="number" stroke="#64748b" fontSize={12} tickFormatter={v => '$' + (v / 1000).toFixed(0) + 'K'} />
                                <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={11} width={90} />
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9' }}
                                    formatter={v => '$' + v.toLocaleString()}
                                />
                                <Bar dataKey="cost" radius={[0, 4, 4, 0]}>
                                    {avgCostData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </motion.div>

                    {/* Feature Importance */}
                    <motion.div className="chart-card glass-card full-width" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <motion.h3 className="chart-title" custom={0} variants={fadeUp}>🔬 Feature Importance (Gradient-Based)</motion.h3>
                        <motion.p className="chart-subtitle" custom={1} variants={fadeUp}>Mean absolute gradient w.r.t. each input feature</motion.p>
                        <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={importanceData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis type="number" stroke="#64748b" fontSize={12} />
                                <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={11} width={200} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#f1f5f9' }} />
                                <Bar dataKey="value" radius={[0, 4, 4, 0]} fill="#06d6f2">
                                    {importanceData.map((_, i) => (
                                        <Cell key={i} fill={`hsl(${180 + i * 8}, 80%, 55%)`} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </motion.div>
                </div>

                {/* Cost Summary */}
                {stats && (
                    <motion.section className="cost-summary" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                        <motion.h2 className="section-title" custom={0} variants={fadeUp}>
                            <span className="icon">📊</span> Cost Distribution Summary
                        </motion.h2>
                        <div className="grid-3">
                            {[
                                { label: 'Minimum Cost', value: '$' + stats.cost.min.toLocaleString(undefined, { maximumFractionDigits: 0 }), badge: 'badge-emerald' },
                                { label: 'Median Cost', value: '$' + stats.cost.median.toLocaleString(undefined, { maximumFractionDigits: 0 }), badge: 'badge-cyan' },
                                { label: 'Maximum Cost', value: '$' + stats.cost.max.toLocaleString(undefined, { maximumFractionDigits: 0 }), badge: 'badge-rose' },
                            ].map((item, i) => (
                                <motion.div className="stat-card" key={item.label} custom={i + 1} variants={fadeUp}>
                                    <div className="stat-value">{item.value}</div>
                                    <div className="stat-label">{item.label}</div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.section>
                )}
            </div>
        </div>
    )
}
