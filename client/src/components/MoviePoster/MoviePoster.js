import { React, useState, useEffect } from "react";
import "./MoviePoster.css";
import axios from 'axios';

const API_KEY = process.env.REACT_APP_TMDB_API_KEY
const API_URL = process.env.REACT_APP_TMDB_API_URL
const IMAGE_URL = process.env.REACT_APP_TMDB_IMAGE_URL
const SIZE = "/w300"

const MoviePoster = ({ movie }) => {
  const [posterURL, setposterURL] = useState(null);

  useEffect(() => {
    if (movie.tmdbId === 0) return;
    axios.get(`${API_URL}/movie/${movie.tmdbId}?api_key=${API_KEY}`)
      .then(res => {
        const imgURL = res.data['poster_path'];
        setposterURL(`${IMAGE_URL}${SIZE}${imgURL}`);
      })
  }, [movie]);
  return (
    <div className="movie-poster">
      <img src={posterURL} alt={`Poster for ${movie.title}`} />
    </div>
  );
};

export default MoviePoster;
