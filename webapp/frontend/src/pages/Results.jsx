import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import './Results.css'

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({ opacity: 1, y: 0, transition: { delay: i * 0.08, duration: 0.5 } })
}

const EVAL_NAMES = {
    '01_loss_curves.png': 'Training & Validation Loss Curves',
    '02_predicted_vs_actual.png': 'Predicted vs Actual Scatter',
    '03_residual_distribution.png': 'Residual Distribution',
    '04_error_by_complexity.png': 'Error by Complexity Level',
    '05_error_by_domain.png': 'Error by Domain',
    '06_cumulative_error.png': 'Cumulative Error Distribution',
    '07_uncertainty_calibration.png': 'Uncertainty Calibration',
    '08_prediction_intervals.png': 'Prediction Intervals (90% CI)',
}

const ANALYSIS_NAMES = {
    'feature_distributions.png': 'Feature Distributions',
    'correlation_heatmap.png': 'Correlation Heatmap',
    'cost_by_category.png': 'Cost by Category',
    'feature_importance.png': 'Feature Importance',
}

export default function Results() {
    const [plots, setPlots] = useState({ eval: [], analysis: [] })
    const [lightbox, setLightbox] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch('/api/plots/list')
            .then(r => r.json())
            .then(p => { setPlots(p); setLoading(false) })
            .catch(() => setLoading(false))
    }, [])

    if (loading) return <div className="page"><div className="container"><div className="spinner" /></div></div>

    return (
        <div className="page results-page">
            <div className="container">
                <motion.div className="page-header" initial="hidden" animate="visible">
                    <motion.h1 custom={0} variants={fadeUp}>Results Gallery</motion.h1>
                    <motion.p custom={1} variants={fadeUp}>
                        13 publication-quality plots — click any image to enlarge
                    </motion.p>
                </motion.div>

                {/* Evaluation Plots */}
                <motion.section initial="hidden" whileInView="visible" viewport={{ once: true }}>
                    <motion.h2 className="section-title" custom={0} variants={fadeUp}>
                        <span className="icon">📊</span> Evaluation Plots
                    </motion.h2>
                    <div className="plots-grid">
                        {plots.eval.map((filename, i) => (
                            <motion.div className="plot-card glass-card" key={filename} custom={i + 1} variants={fadeUp}>
                                <img
                                    src={`/api/plots/eval/${filename}`}
                                    alt={EVAL_NAMES[filename] || filename}
                                    className="plot-image"
                                    onClick={() => setLightbox(`/api/plots/eval/${filename}`)}
                                    loading="lazy"
                                />
                                <p className="plot-label">{EVAL_NAMES[filename] || filename.replace('.png', '').replace(/_/g, ' ')}</p>
                            </motion.div>
                        ))}
                    </div>
                </motion.section>

                {/* Analysis Plots */}
                <motion.section className="analysis-section" initial="hidden" whileInView="visible" viewport={{ once: true }}>
                    <motion.h2 className="section-title" custom={0} variants={fadeUp}>
                        <span className="icon">🔬</span> Dataset Analysis Plots
                    </motion.h2>
                    <div className="plots-grid">
                        {plots.analysis.filter(f => f.endsWith('.png')).map((filename, i) => (
                            <motion.div className="plot-card glass-card" key={filename} custom={i + 1} variants={fadeUp}>
                                <img
                                    src={`/api/plots/analysis/${filename}`}
                                    alt={ANALYSIS_NAMES[filename] || filename}
                                    className="plot-image"
                                    onClick={() => setLightbox(`/api/plots/analysis/${filename}`)}
                                    loading="lazy"
                                />
                                <p className="plot-label">{ANALYSIS_NAMES[filename] || filename.replace('.png', '').replace(/_/g, ' ')}</p>
                            </motion.div>
                        ))}
                    </div>
                </motion.section>
            </div>

            {/* Lightbox */}
            <AnimatePresence>
                {lightbox && (
                    <motion.div
                        className="modal-overlay"
                        onClick={() => setLightbox(null)}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <motion.img
                            src={lightbox}
                            alt="Enlarged plot"
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.8, opacity: 0 }}
                            transition={{ duration: 0.3 }}
                        />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
