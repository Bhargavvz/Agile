import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Predict from './pages/Predict'
import About from './pages/About'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/predict" element={<Predict />} />
      <Route path="/about" element={<About />} />
    </Routes>
  )
}
