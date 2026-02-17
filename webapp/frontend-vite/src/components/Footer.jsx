import { Link } from "react-router-dom";

export default function Footer() {
    return (
        <footer className="mt-auto border-t border-white/5">
            <div className="max-w-7xl mx-auto px-6 py-12">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
                    {/* Brand */}
                    <div>
                        <div className="flex items-center gap-3 mb-4">
                            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                                    <path d="M2 17l10 5 10-5" />
                                    <path d="M2 12l10 5 10-5" />
                                </svg>
                            </div>
                            <span className="text-lg font-bold gradient-text">AgileScope AI</span>
                        </div>
                        <p className="text-sm text-slate-500 leading-relaxed max-w-xs">
                            Deep learning-powered cost estimation for Agile software projects with uncertainty quantification.
                        </p>
                    </div>

                    {/* Links */}
                    <div>
                        <h4 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider">Navigation</h4>
                        <div className="flex flex-col gap-2">
                            <Link to="/" className="text-sm text-slate-500 hover:text-violet-400 transition-colors">Home</Link>
                            <Link to="/predict" className="text-sm text-slate-500 hover:text-violet-400 transition-colors">Cost Estimator</Link>
                            <Link to="/about" className="text-sm text-slate-500 hover:text-violet-400 transition-colors">About the Model</Link>
                        </div>
                    </div>

                    {/* Tech */}
                    <div>
                        <h4 className="text-sm font-semibold text-slate-300 mb-4 uppercase tracking-wider">Technology</h4>
                        <div className="flex flex-wrap gap-2">
                            {["BERT-Large", "PyTorch", "FastAPI", "React + Vite", "MC Dropout"].map((t) => (
                                <span key={t} className="px-3 py-1 rounded-full text-xs font-medium bg-white/5 text-slate-400 border border-white/8">
                                    {t}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="mt-10 pt-6 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
                    <p className="text-xs text-slate-600">
                        © 2026 AgileScope AI. Built with BERT-Large + MC Dropout Uncertainty.
                    </p>
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                        <span className="text-xs text-slate-500">Model Active</span>
                    </div>
                </div>
            </div>
        </footer>
    );
}
