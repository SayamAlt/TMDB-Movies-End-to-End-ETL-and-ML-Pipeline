# TMDB Movies ETL and ML Project

This project focuses on performing end-to-end data processing, exploratory analysis, and machine learning using data from **The Movie Database (TMDB)**. The project is structured to showcase skills in data ingestion, transformation, analysis, and predictive modeling, with the ultimate goal of deriving insights and building a regression model to predict average movie ratings.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Model](#machine-learning-model)
- [Results](#results)
- [Setup and Execution](#setup-and-execution)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## Introduction

The TMDB Movies ETL and ML Project demonstrates the integration of ETL (Extract, Transform, Load) processes with machine learning. By utilizing the TMDB API, multiple datasets related to movies were ingested and combined. The processed data underwent exploratory data analysis to uncover meaningful insights. Finally, a regression model was built to predict average movie ratings with a high degree of accuracy.

---

## Features

- **Data Ingestion**: Automated extraction of data from TMDB API for various categories (Top Rated, Popular, Upcoming, Current Movies, etc.).
- **Data Transformation**: Merging datasets with genre information and preprocessing for analysis.
- **Exploratory Data Analysis**: Statistical and visual insights into movie trends, ratings, popularity, and genres.
- **Machine Learning**: Built a regression model to predict average movie ratings with an R² score of 97%.
- **Reusable Pipeline**: A modular and scalable approach to ETL and modeling.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Requests
- **APIs**: TMDB API
- **Environment**: Jupyter Notebook
- **Version Control**: Git
- **Machine Learning**: Regression Models

---

## Data Processing Pipeline

1. **Extraction**:
   - Fetch data from TMDB API for multiple endpoints such as:
     - Top Rated Movies
     - Popular Movies
     - Upcoming Movies
     - Current Movies
     - Genre Information
2. **Transformation**:
   - Combine all movie datasets.
   - Map genre IDs to their respective names.
   - Feature engineering for better insights and model readiness (e.g., `release_year`, `movie_age`).
   - Normalize continuous variables like popularity.
3. **Loading**:
   - Store the processed data in a structured format for further analysis and machine learning.

---

## Exploratory Data Analysis (EDA)

Insights uncovered during EDA include:
- Top genres based on the count of movies.
- Correlation between popularity and average rating.
- Trends in movie production over the years.
- Patterns in movie length, release times, and genre popularity.

Visualizations were created to represent:
- Distribution of movie ratings.
- Popularity trends over time.
- Genre-based segmentation of movies.

---

## Machine Learning Model

- **Goal**: Predict the average rating (`vote_average`) of a movie.
- **Model**: Regression model using Scikit-learn.
- **Features Used**:
  - Popularity
  - Genre
  - Release Year
  - Movie Age
- **Performance**:
  - Achieved an R² score of 97%.
  - Model tested and validated using multiple evaluation metrics.

---

## Results

- Successfully built a robust data processing pipeline.
- Conducted detailed EDA to derive actionable insights.
- Developed a high-performing regression model for movie rating prediction.
- Achieved clear and actionable results through statistical and machine learning approaches.

---

## Setup and Execution

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
2. Install Dependencies

```bash
pip install -r requirements.txt

### Set Up TMDB API Key

1. Create an account on [TMDB](https://www.themoviedb.org/).
2. Generate an API key.
3. Save the API key as an environment variable:

   - **On Windows**:
     ```bash
     set TMDB_API_KEY=<your_api_key>
     ```

   - **On macOS/Linux**:
     ```bash
     export TMDB_API_KEY=<your_api_key>
     ```

### Run the Pipeline

1. Open the provided Jupyter Notebook.
2. Execute the ETL process and machine learning model step-by-step.

---

### Future Improvements

- Add support for real-time data streaming.
- Integrate additional data sources for richer insights (e.g., IMDb, Rotten Tomatoes).
- Build a more complex predictive model incorporating ensemble techniques.
- Develop a user-friendly dashboard for real-time movie analytics.

---

### Acknowledgments

- **TMDB**: For providing the API and movie datasets.
- **Open-source Tools**: For enabling seamless data processing and machine learning.

