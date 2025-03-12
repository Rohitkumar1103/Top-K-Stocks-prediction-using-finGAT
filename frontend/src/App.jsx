import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch data when component mounts
    fetchStockRecommendations();
  }, []);

  const fetchStockRecommendations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:3001/api/top-stocks', {});
      setStocks(response.data.top_stocks);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch stock recommendations. ' + (err.response?.data?.error || err.message));
      setLoading(false);
      console.error('Error fetching stock data:', err);
    }
  };

  // Function to format return values as percentages
  const formatReturn = (value) => {
    return (value * 100).toFixed(12) + '%';
  };

  // Function to determine the color based on return value
  const getReturnColor = (value) => {
    return value >= 0 ? 'green' : 'red';
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Stock Recommendations</h1>
        <button 
          className="refresh-button" 
          onClick={fetchStockRecommendations}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </header>

      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={fetchStockRecommendations}>Try Again</button>
        </div>
      )}

      {loading ? (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading recommendations...</p>
        </div>
      ) : (
        <div className="stocks-container">
          {stocks.length > 0 ? (
            <>
              <div className="stocks-header">
                <h2>Top 5 Recommended Stocks</h2>
                <p>Based on predicted returns</p>
              </div>
              <div className="stocks-grid">
                {stocks.map((stock, index) => (
                  <div className="stock-card" key={index}>
                    <div className="stock-symbol">{stock.symbol}</div>
                    <div className="stock-sector">{stock.sector}</div>
                    <div 
                      className="stock-return"
                      style={{ color: getReturnColor(stock.predicted_return) }}
                    >
                      {formatReturn(stock.predicted_return)}
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="no-stocks">
              <p>No stock recommendations available.</p>
            </div>
          )}
        </div>
      )}

      <footer className="app-footer">
        <p>Financial Stock Recommender &copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;
