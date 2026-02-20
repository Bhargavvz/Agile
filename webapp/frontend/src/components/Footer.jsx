import './Footer.css'

export default function Footer() {
    return (
        <footer className="footer" id="footer">
            <div className="container">
                <div className="footer-inner">
                    <div className="footer-brand">
                        <span className="brand-icon">⚡</span>
                        <span>AgileCost<span className="brand-highlight">AI</span></span>
                    </div>
                    <p className="footer-desc">
                        Software Cost Estimation in Agile Methodology using Deep Learning
                    </p>
                    <div className="footer-bottom">
                        <span className="footer-tech">BERT-Large · PyTorch · FastAPI · React</span>
                        <span className="footer-copy">© 2026 Research Project</span>
                    </div>
                </div>
            </div>
        </footer>
    )
}
