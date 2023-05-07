import React from "react";
import "./MovieList.css";
import MovieCard from "../MovieCard/MovieCard";

const MovieList = ({ items, onClickAction }) => {
  return (
    <div className="movie-card-list">
      {items.map((movie, index) => (
        <MovieCard
          key={index}
          movie={movie}
          onClick={() => { onClickAction(index, items) }} />
      ))}
    </div>
  );
};

export default MovieList;


