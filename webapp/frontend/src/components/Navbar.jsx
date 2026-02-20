import { NavLink } from 'react-router-dom'
import { useState } from 'react'
import './Navbar.css'

const links = [
    { to: '/', label: 'Home' },
    { to: '/dashboard', label: 'Dashboard' },
    { to: '/predict', label: 'Predict' },
    { to: '/dataset', label: 'Dataset' },
    { to: '/results', label: 'Results' },
    { to: '/about', label: 'About' },
]

export default function Navbar() {
    const [menuOpen, setMenuOpen] = useState(false)

    return (
        <nav className="navbar" id="main-nav">
            <div className="navbar-inner container">
                <NavLink to="/" className="navbar-brand">
                    <span className="brand-icon">⚡</span>
                    <span className="brand-text">AgileCost<span className="brand-highlight">AI</span></span>
                </NavLink>

                <button
                    className={`navbar-toggle ${menuOpen ? 'active' : ''}`}
                    onClick={() => setMenuOpen(!menuOpen)}
                    aria-label="Toggle menu"
                >
                    <span /><span /><span />
                </button>

                <div className={`navbar-links ${menuOpen ? 'open' : ''}`}>
                    {links.map(({ to, label }) => (
                        <NavLink
                            key={to}
                            to={to}
                            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                            onClick={() => setMenuOpen(false)}
                            end={to === '/'}
                        >
                            {label}
                        </NavLink>
                    ))}
                </div>
            </div>
        </nav>
    )
}
