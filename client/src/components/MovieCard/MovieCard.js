import React, { useState } from "react";
import "./MovieCard.css";
import MoviePoster from "../MoviePoster/MoviePoster";


const MovieCard = ({ movie, onClick }) => {
  const [isHovering, setIsHovering] = useState(false);

  const handleMouseEnter = () => {
    setIsHovering(true);
  };

  const handleMouseLeave = () => {
    setIsHovering(false);
  };



  return (
    <div
      className="movie-card"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
    >
      <MoviePoster movie={movie} />
      {isHovering && (
        <div className="movie-details">
          <h2>{movie.title} ({movie.year})</h2>
          <p>{movie.genres}</p>
        </div>
      )}
    </div>
  );
};

export default MovieCard;