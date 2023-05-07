import React, { useState } from 'react';
import axios from 'axios';
import './Search.css';
import MovieCard from '../MovieCard/MovieCard';

function Search({ handleSearchResults, onClickAction }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [showResults, setShowResults] = useState(false);

  const handleSearch = async () => {
    if (!searchTerm) return;

    try {
      const response = await axios.get(`http://localhost:5000/search?term=${searchTerm}`);
      setSearchResults(response.data);
      setShowResults(true);
    } catch (error) {
      console.error(error);
    }
  };

  const renderResults = () => {
    return (
      <div className="search-results">
        {searchResults.map((movie, index) => (
          <div className="search-result-item">
            <MovieCard
              className='search-result-item'
              key={index}
              movie={movie}
              onClick={() => { onClickAction(index, searchResults) }}
            />
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="search-container">
      <input
        className="search-input"
        type="text"
        placeholder="Search..."
        value={searchTerm}
        onChange={(event) => {
          setSearchTerm(event.target.value);
          setShowResults(false);
        }}
      />
      <button className="search-button" onClick={handleSearch}>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
      </button>
      {showResults && renderResults()}
    </div>
  );
}

export default Search;
