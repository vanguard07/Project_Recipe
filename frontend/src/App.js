// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/" element={<Layout />} />
          <Route path="/chat/:chatId" element={<Layout />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;