import { useState } from 'react'
import { motion } from 'framer-motion'
import './Predict.css'

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1, duration: 0.5 } })
}

export default function Predict() {
    const [reportText, setReportText] = useState('')
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [fileName, setFileName] = useState(null)

    const handleFileUpload = (e) => {
        const file = e.target.files[0]
        if (!file) return
        setFileName(file.name)
        const reader = new FileReader()
        reader.onload = (ev) => setReportText(ev.target.result)
        reader.readAsText(file)
    }

    const loadSampleReport = async () => {
        try {
            const res = await fetch('/api/predict/sample-report')
            const data = await res.json()
            setReportText(data.text)
            setFileName(data.filename)
        } catch {
            setError('Failed to load sample report')
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (reportText.trim().length < 50) {
            setError('Report text is too short. Please paste a full project report (minimum 50 characters).')
            return
        }
        setLoading(true)
        setError(null)
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ report_text: reportText }),
            })
            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Prediction failed')
            }
            const data = await res.json()
            setResult(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const formatUSD = (v) => '$' + v.toLocaleString(undefined, { maximumFractionDigits: 0 })

    return (
        <div className="page predict-page">
            <div className="container">
                <motion.div className="page-header" initial="hidden" animate="visible">
                    <motion.h1 custom={0} variants={fadeUp}>Cost Prediction</motion.h1>
                    <motion.p custom={1} variants={fadeUp}>
                        Upload or paste your Agile project report — the trained BERT-Large model will predict the development cost
                    </motion.p>
                </motion.div>

                <div className="predict-layout">
                    {/* Input Section */}
                    <motion.form className="predict-form glass-card" onSubmit={handleSubmit} initial="hidden" animate="visible">
                        <motion.div className="form-header" custom={0} variants={fadeUp}>
                            <h3>📄 Project Report</h3>
                            <div className="form-actions-top">
                                <label className="btn btn-outline upload-btn" htmlFor="file-upload">
                                    📎 Upload .txt
                                    <input
                                        type="file"
                                        id="file-upload"
                                        accept=".txt"
                                        onChange={handleFileUpload}
                                        style={{ display: 'none' }}
                                    />
                                </label>
                                <button type="button" className="btn btn-outline" onClick={loadSampleReport}>
                                    📋 Load Sample
                                </button>
                            </div>
                        </motion.div>

                        {fileName && (
                            <motion.div className="file-badge" custom={1} variants={fadeUp}>
                                <span className="badge badge-cyan">📎 {fileName}</span>
                            </motion.div>
                        )}

                        <motion.div className="form-group" custom={2} variants={fadeUp}>
                            <label htmlFor="report-text">Paste your project report below</label>
                            <textarea
                                id="report-text"
                                className="report-textarea"
                                value={reportText}
                                onChange={(e) => setReportText(e.target.value)}
                                placeholder={`Paste your Agile project report here...\n\nExample format:\n========================================================\n  PROJECT REPORT — project_00001\n  Domain        : E-commerce\n  Complexity    : Medium\n  Team Size     : 8    Duration : 6 months\n========================================================\n1. Project Overview\n...\n2. Agile Framework\n...\n3. Product Backlog & User Stories\n...\n4. Sprint Planning & Velocity\n...\n5. System Architecture & Modules\n...\n6. Technology Stack\n...\n7. Risk & Requirement Volatility\n...`}
                                rows={18}
                            />
                        </motion.div>

                        <motion.div className="char-count" custom={3} variants={fadeUp}>
                            {reportText.length.toLocaleString()} characters
                            {reportText.length > 0 && reportText.length < 50 && (
                                <span className="char-warn"> (minimum 50 required)</span>
                            )}
                        </motion.div>

                        <motion.button
                            type="submit"
                            className="btn btn-primary submit-btn"
                            disabled={loading || reportText.trim().length < 50}
                            custom={4}
                            variants={fadeUp}
                        >
                            {loading ? (
                                <>
                                    <span className="btn-spinner" /> Analyzing with BERT...
                                </>
                            ) : (
                                '🚀 Predict Cost'
                            )}
                        </motion.button>
                    </motion.form>

                    {/* Result Section */}
                    <div className="predict-result-area">
                        {error && <div className="error-msg">{error}</div>}

                        {result && (
                            <motion.div
                                className="result-card glass-card"
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ duration: 0.4 }}
                            >
                                <h3>💰 Prediction Result</h3>

                                {/* Model badge */}
                                <div className="model-badge">
                                    <span className={`badge ${result.model_used.includes('BERT') ? 'badge-emerald' : 'badge-amber'}`}>
                                        {result.model_used.includes('BERT') ? '🧠 ' : '📐 '}{result.model_used}
                                    </span>
                                    {result.mc_passes > 0 && (
                                        <span className="badge badge-violet">MC×{result.mc_passes}</span>
                                    )}
                                    <span className="badge badge-cyan">{result.device.toUpperCase()}</span>
                                </div>

                                {/* Main cost */}
                                <div className="result-main">
                                    <div className="result-cost">{formatUSD(result.predicted_cost)}</div>
                                    <div className="result-label">Estimated Project Cost</div>
                                </div>

                                {/* Uncertainty details */}
                                <div className="result-details">
                                    <div className="result-row">
                                        <span>Aleatoric Uncertainty</span>
                                        <span className="mono">±{formatUSD(result.aleatoric_uncertainty)}</span>
                                    </div>
                                    <div className="result-row">
                                        <span>Epistemic Uncertainty</span>
                                        <span className="mono">±{formatUSD(result.epistemic_uncertainty)}</span>
                                    </div>
                                    <div className="result-row highlight">
                                        <span>Total Uncertainty</span>
                                        <span className="mono">±{formatUSD(result.total_uncertainty)}</span>
                                    </div>
                                </div>

                                {/* Confidence Interval */}
                                <div className="ci-section">
                                    <h4>90% Confidence Interval</h4>
                                    <div className="ci-bar">
                                        <div className="ci-range">
                                            <span>{formatUSD(result.confidence_interval_low)}</span>
                                            <span className="ci-center">{formatUSD(result.predicted_cost)}</span>
                                            <span>{formatUSD(result.confidence_interval_high)}</span>
                                        </div>
                                        <div className="confidence-bar">
                                            <div className="confidence-bar-fill" style={{ width: '100%' }} />
                                            <div className="confidence-bar-marker" style={{ left: '50%' }} />
                                        </div>
                                    </div>
                                </div>

                                {/* Extracted features */}
                                {result.parsed_features && (
                                    <div className="extracted-section">
                                        <h4>📋 Extracted from Report</h4>
                                        <div className="extracted-grid">
                                            <div className="extracted-item">
                                                <span className="ex-label">Domain</span>
                                                <span className="ex-value">{result.parsed_features.domain}</span>
                                            </div>
                                            <div className="extracted-item">
                                                <span className="ex-label">Complexity</span>
                                                <span className="ex-value">{result.parsed_features.complexity_level}</span>
                                            </div>
                                            <div className="extracted-item">
                                                <span className="ex-label">Tech Stack</span>
                                                <span className="ex-value">{result.parsed_features.tech_stack_difficulty}</span>
                                            </div>
                                            <div className="extracted-item">
                                                <span className="ex-label">Risk Level</span>
                                                <span className="ex-value">{result.parsed_features.risk_level}</span>
                                            </div>
                                            <div className="extracted-item">
                                                <span className="ex-label">Volatility</span>
                                                <span className="ex-value">{result.parsed_features.requirement_volatility}</span>
                                            </div>
                                            {result.parsed_features.numeric && Object.entries(result.parsed_features.numeric).map(([k, v]) => (
                                                <div className="extracted-item" key={k}>
                                                    <span className="ex-label">{k.replace(/_/g, ' ')}</span>
                                                    <span className="ex-value">{v}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </motion.div>
                        )}

                        {!result && !error && (
                            <div className="result-placeholder glass-card">
                                <div className="placeholder-icon">📄</div>
                                <h3>Upload Your Project Report</h3>
                                <p>
                                    Paste or upload an Agile project report (like the synthetic reports in the dataset).
                                    The trained BERT-Large model will analyse the text and predict the development cost
                                    with uncertainty quantification.
                                </p>
                                <div className="placeholder-features">
                                    <div className="pf-item"><span>📝</span> Natural language report analysis</div>
                                    <div className="pf-item"><span>🧠</span> BERT-Large (340M params)</div>
                                    <div className="pf-item"><span>🔬</span> MC Dropout uncertainty (20 passes)</div>
                                    <div className="pf-item"><span>📊</span> 90% confidence interval</div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
