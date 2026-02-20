import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import Dataset from './pages/Dataset'
import Results from './pages/Results'
import About from './pages/About'

export default function App() {
  return (
    <>
      <div className="bg-glow bg-glow-cyan" />
      <div className="bg-glow bg-glow-violet" />
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/results" element={<Results />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <Footer />
    </>
  )
}
