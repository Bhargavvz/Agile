import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";

const navLinks = [
    { to: "/", label: "Home" },
    { to: "/predict", label: "Predict" },
    { to: "/about", label: "About" },
];

export default function Navbar() {
    const { pathname } = useLocation();
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const h = () => setScrolled(window.scrollY > 20);
        window.addEventListener("scroll", h);
        return () => window.removeEventListener("scroll", h);
    }, []);

    return (
        <motion.nav
            initial={{ y: -80, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${scrolled ? "glass-strong shadow-2xl py-3" : "bg-transparent py-5"
                }`}
        >
            <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
                {/* Logo */}
                <Link to="/" className="flex items-center gap-3 group">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-lg group-hover:shadow-violet-500/30 transition-shadow duration-300">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 2L2 7l10 5 10-5-10-5z" />
                            <path d="M2 17l10 5 10-5" />
                            <path d="M2 12l10 5 10-5" />
                        </svg>
                    </div>
                    <span className="text-xl font-bold gradient-text tracking-tight">AgileScope AI</span>
                </Link>

                {/* Links */}
                <div className="hidden md:flex items-center gap-1">
                    {navLinks.map((l) => {
                        const active = pathname === l.to;
                        return (
                            <Link
                                key={l.to}
                                to={l.to}
                                className={`relative px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${active ? "text-white" : "text-slate-400 hover:text-white"
                                    }`}
                            >
                                {active && (
                                    <motion.div
                                        layoutId="nav"
                                        className="absolute inset-0 rounded-xl bg-white/10 border border-white/10"
                                        transition={{ type: "spring", stiffness: 400, damping: 30 }}
                                    />
                                )}
                                <span className="relative z-10">{l.label}</span>
                            </Link>
                        );
                    })}
                </div>

                {/* CTA */}
                <Link
                    to="/predict"
                    className="hidden md:flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 text-white text-sm font-semibold hover:from-violet-500 hover:to-indigo-500 transition-all duration-300 shadow-lg hover:shadow-violet-500/25"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="1" x2="12" y2="23" />
                        <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                    </svg>
                    Estimate Cost
                </Link>

                {/* Mobile */}
                <div className="md:hidden flex items-center gap-2">
                    {navLinks.map((l) => (
                        <Link key={l.to} to={l.to} className={`px-3 py-2 rounded-lg text-xs font-medium ${pathname === l.to ? "bg-white/10 text-white" : "text-slate-400"}`}>
                            {l.label}
                        </Link>
                    ))}
                </div>
            </div>
        </motion.nav>
    );
}
