import logo from './logo.svg';
import './App.css';
import MovieList from './components/MovieList/MovieList';
import { useEffect, useState } from 'react';
import Search from './components/Search/Search';
import axios from 'axios';
function App() {
  const [recommendations, setRecommendations] = useState([]);
  const [userPreference, setUserPreference] = useState([]);
  const [searchResults, setSearchResults] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:5000/sample')
      .then(res => res.data)
      .then(data => setRecommendations(data))
  }, []);

  useEffect(() => {
    if ((userPreference.length < 3))
      return;
    axios.post("http://localhost:5000/recommend", {
      userPreference
    })
      .then(res => { console.log(res.data); return res.data })
      .then(data => setRecommendations(data))
    return () => {

    };
  }, [userPreference]);


  const addToPreferences = (index, list) => {
    const movie = list[index];
    setUserPreference([...userPreference, movie]);
  }

  const removeFromPreferences = (index, list) => {
    setUserPreference(list.filter((_, i) => i !== index))
  }

  return (
    <div className="App">
      <Search onClickAction={addToPreferences}></Search>
      <MovieList id='recommendations' items={recommendations} onClickAction={addToPreferences}></MovieList>
      <MovieList id='preferences' items={userPreference} onClickAction={removeFromPreferences}></MovieList>
    </div>
  );
}

export default App;
