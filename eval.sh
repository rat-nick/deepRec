#!/bin/bash
nohup python -m recs.evaluation --user-threshold 20 --ratings-path data/clean/ratings.csv --result-path recres_20.csv --leave-one-out --ranking &
nohup python -m recs.evaluation --user-threshold 50 --ratings-path data/clean/ratings.csv --result-path recres_50.csv --leave-one-out --ranking &
nohup python -m recs.evaluation --user-threshold 100 --ratings-path data/clean/ratings.csv --result-path recres_100.csv --leave-one-out --ranking &
nohup python -m recs.evaluation --user-threshold 200 --ratings-path data/clean/ratings.csv --result-path recres_200.csv --leave-one-out --ranking &