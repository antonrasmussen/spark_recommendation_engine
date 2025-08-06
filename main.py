# MovieLens Recommendation System Implementation Guide
# Using Apache Spark MLlib with ALS Algorithm

# ============================================================================
# PHASE 1: ENVIRONMENT SETUP AND DATA EXPLORATION
# ============================================================================

"""
Step 1: Install Required Libraries
!pip install pyspark
!pip install pandas matplotlib seaborn
"""

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, count, avg, max, min, stddev
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize Spark Session
def create_spark_session():
    """
    Create Spark session with optimized configuration for recommendation systems
    """
    spark = SparkSession.builder \
        .appName("MovieLens Recommendation System") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

# ============================================================================
# PHASE 2: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(spark):
    """
    Load MovieLens dataset files
    Note: Since you have split files, we'll need to combine them
    """
    
    # Load movies data
    movies_schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True)
    ])
    
    movies = spark.read.csv("movies.csv", header=True, schema=movies_schema)
    
    # Load and combine all rating files (you have multiple splits)
    rating_files = [
        "ratings_split_1.csv", "ratings_split_2.csv", "ratings_split_3.csv",
        # Add all your rating split files here
    ]
    
    ratings_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", DoubleType(), True),
        StructField("timestamp", LongType(), True)
    ])
    
    # Read first file
    ratings = spark.read.csv(rating_files[0], header=True, schema=ratings_schema)
    
    # Union with remaining files
    for file in rating_files[1:]:
        temp_df = spark.read.csv(file, header=True, schema=ratings_schema)
        ratings = ratings.union(temp_df)
    
    # Load tags (optional - for content-based features later)
    tags_files = ["tags_split_1.csv", "tags_split_2.csv", "tags_split_3.csv", "tags_split_4.csv"]
    
    tags_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("tag", StringType(), True),
        StructField("timestamp", LongType(), True)
    ])
    
    tags = spark.read.csv(tags_files[0], header=True, schema=tags_schema)
    for file in tags_files[1:]:
        temp_df = spark.read.csv(file, header=True, schema=tags_schema)
        tags = tags.union(temp_df)
    
    return movies, ratings, tags

def explore_data(movies, ratings, tags):
    """
    Comprehensive data exploration for presentation
    """
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    
    # Basic statistics
    print(f"Number of movies: {movies.count():,}")
    print(f"Number of ratings: {ratings.count():,}")
    print(f"Number of tags: {tags.count():,}")
    print(f"Number of unique users: {ratings.select('userId').distinct().count():,}")
    
    # Rating distribution
    print("\nRating Distribution:")
    ratings.groupBy("rating").count().orderBy("rating").show()
    
    # User activity statistics
    user_stats = ratings.groupBy("userId").agg(
        count("*").alias("num_ratings"),
        avg("rating").alias("avg_rating")
    )
    
    print("\nUser Activity Statistics:")
    user_stats.agg(
        avg("num_ratings").alias("avg_ratings_per_user"),
        min("num_ratings").alias("min_ratings"),
        max("num_ratings").alias("max_ratings"),
        stddev("num_ratings").alias("std_ratings")
    ).show()
    
    # Movie popularity statistics
    movie_stats = ratings.groupBy("movieId").agg(
        count("*").alias("num_ratings"),
        avg("rating").alias("avg_rating")
    )
    
    print("\nMovie Popularity Statistics:")
    movie_stats.agg(
        avg("num_ratings").alias("avg_ratings_per_movie"),
        min("num_ratings").alias("min_ratings"),
        max("num_ratings").alias("max_ratings"),
        stddev("num_ratings").alias("std_ratings")
    ).show()
    
    return user_stats, movie_stats

# ============================================================================
# PHASE 3: DATA PREPROCESSING
# ============================================================================

def preprocess_data(ratings):
    """
    Clean and prepare data for model training
    """
    print("Preprocessing data...")
    
    # Remove any null values
    clean_ratings = ratings.filter(
        col("userId").isNotNull() & 
        col("movieId").isNotNull() & 
        col("rating").isNotNull()
    )
    
    # Filter out users and movies with very few ratings (cold start mitigation)
    # Users with at least 20 ratings (dataset already filtered for this)
    user_counts = clean_ratings.groupBy("userId").count()
    active_users = user_counts.filter(col("count") >= 20).select("userId")
    
    # Movies with at least 10 ratings
    movie_counts = clean_ratings.groupBy("movieId").count()
    popular_movies = movie_counts.filter(col("count") >= 10).select("movieId")
    
    # Filter ratings to include only active users and popular movies
    filtered_ratings = clean_ratings.join(active_users, "userId") \
                                  .join(popular_movies, "movieId")
    
    print(f"Original ratings: {ratings.count():,}")
    print(f"After preprocessing: {filtered_ratings.count():,}")
    
    return filtered_ratings

# ============================================================================
# PHASE 4: MODEL TRAINING AND EVALUATION
# ============================================================================

def train_als_model(ratings):
    """
    Train ALS recommendation model with hyperparameter tuning
    """
    print("Training ALS model...")
    
    # Split data into train and test
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=42)
    
    # Cache datasets for better performance
    training.cache()
    test.cache()
    
    print(f"Training set size: {training.count():,}")
    print(f"Test set size: {test.count():,}")
    
    # Configure ALS algorithm
    als = ALS(
        maxIter=10,                    # Number of iterations
        regParam=0.01,                 # Regularization parameter
        userCol="userId",
        itemCol="movieId", 
        ratingCol="rating",
        coldStartStrategy="drop",      # Drop predictions for unknown users/items
        nonnegative=True,             # Ensure non-negative factors
        rank=50                       # Number of latent factors
    )
    
    # Train the model
    model = als.fit(training)
    
    return model, training, test

def evaluate_model(model, test):
    """
    Evaluate model performance using multiple metrics
    """
    print("Evaluating model...")
    
    # Generate predictions
    predictions = model.transform(test)
    
    # Remove NaN predictions (from cold start)
    predictions = predictions.filter(col("prediction").isNotNull())
    
    # Calculate RMSE
    rmse_evaluator = RegressionEvaluator(
        metricName="rmse", 
        labelCol="rating", 
        predictionCol="prediction"
    )
    rmse = rmse_evaluator.evaluate(predictions)
    
    # Calculate MAE
    mae_evaluator = RegressionEvaluator(
        metricName="mae", 
        labelCol="rating", 
        predictionCol="prediction"
    )
    mae = mae_evaluator.evaluate(predictions)
    
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    return predictions, rmse, mae

# ============================================================================
# PHASE 5: RECOMMENDATION GENERATION
# ============================================================================

def generate_recommendations(model, ratings, num_recommendations=10):
    """
    Generate top-N recommendations for all users
    """
    print(f"Generating top-{num_recommendations} recommendations for all users...")
    
    # Generate recommendations for all users
    user_recs = model.recommendForAllUsers(num_recommendations)
    
    # Generate recommendations for all movies (items)
    movie_recs = model.recommendForAllItems(num_recommendations)
    
    print("Sample user recommendations:")
    user_recs.show(5, truncate=False)
    
    return user_recs, movie_recs

def get_movie_recommendations_for_user(user_id, model, movies, num_recommendations=10):
    """
    Get readable movie recommendations for a specific user
    """
    from pyspark.sql import Row
    
    # Create a dataframe with single user
    user_df = spark.createDataFrame([Row(userId=user_id)])
    
    # Get recommendations
    user_recs = model.recommendForUserSubset(user_df, num_recommendations)
    
    # Extract movie IDs and ratings from recommendations
    recs_with_movies = user_recs.select(
        col("userId"),
        col("recommendations.movieId").alias("movieIds"),
        col("recommendations.rating").alias("predicted_ratings")
    )
    
    return recs_with_movies

# ============================================================================
# PHASE 6: HYPERPARAMETER TUNING (OPTIONAL)
# ============================================================================

def hyperparameter_tuning(ratings):
    """
    Perform hyperparameter tuning using cross-validation
    """
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    
    print("Performing hyperparameter tuning...")
    
    # Split data
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=42)
    
    # Create ALS model
    als = ALS(
        userCol="userId",
        itemCol="movieId", 
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )
    
    # Create parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(als.rank, [10, 25, 50]) \
        .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
        .addGrid(als.maxIter, [5, 10, 15]) \
        .build()
    
    # Create evaluator
    evaluator = RegressionEvaluator(
        metricName="rmse", 
        labelCol="rating", 
        predictionCol="prediction"
    )
    
    # Create cross-validator
    crossval = CrossValidator(
        estimator=als,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3  # 3-fold cross-validation
    )
    
    # Run cross-validation
    cv_model = crossval.fit(training)
    
    # Get best model
    best_model = cv_model.bestModel
    
    print(f"Best rank: {best_model.rank}")
    print(f"Best regParam: {best_model._java_obj.regParam()}")
    print(f"Best maxIter: {best_model._java_obj.maxIter()}")
    
    return best_model

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline for the recommendation system
    """
    # Initialize Spark
    spark = create_spark_session()
    
    try:
        # Phase 1: Load data
        print("Loading data...")
        movies, ratings, tags = load_data(spark)
        
        # Phase 2: Explore data
        user_stats, movie_stats = explore_data(movies, ratings, tags)
        
        # Phase 3: Preprocess data
        clean_ratings = preprocess_data(ratings)
        
        # Phase 4: Train model
        model, training, test = train_als_model(clean_ratings)
        
        # Phase 5: Evaluate model
        predictions, rmse, mae = evaluate_model(model, test)
        
        # Phase 6: Generate recommendations
        user_recs, movie_recs = generate_recommendations(model, clean_ratings)
        
        # Phase 7: Sample specific user recommendations
        sample_user_id = ratings.select("userId").first()[0]
        sample_recs = get_movie_recommendations_for_user(
            sample_user_id, model, movies, 10
        )
        
        print("=" * 50)
        print("MODEL TRAINING COMPLETE!")
        print(f"Final RMSE: {rmse:.4f}")
        print(f"Final MAE: {mae:.4f}")
        print("=" * 50)
        
        # Save model (optional)
        # model.write().overwrite().save("movielen_als_model")
        
        return model, predictions, user_recs, movie_recs
        
    finally:
        # Clean up
        spark.stop()

# ============================================================================
# VISUALIZATION FUNCTIONS FOR PRESENTATION
# ============================================================================

def create_visualizations(ratings_pd, movies_pd):
    """
    Create visualizations for PowerPoint presentation
    Note: Convert Spark DataFrames to Pandas for visualization
    """
    
    # Rating distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    ratings_pd['rating'].hist(bins=10, edgecolor='black')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    
    # User activity distribution
    plt.subplot(2, 2, 2)
    user_activity = ratings_pd.groupby('userId').size()
    user_activity.hist(bins=50, edgecolor='black')
    plt.title('User Activity Distribution')
    plt.xlabel('Number of Ratings per User')
    plt.ylabel('Number of Users')
    plt.yscale('log')
    
    # Movie popularity distribution
    plt.subplot(2, 2, 3)
    movie_popularity = ratings_pd.groupby('movieId').size()
    movie_popularity.hist(bins=50, edgecolor='black')
    plt.title('Movie Popularity Distribution')
    plt.xlabel('Number of Ratings per Movie')
    plt.ylabel('Number of Movies')
    plt.yscale('log')
    
    # Genre distribution
    plt.subplot(2, 2, 4)
    genres = movies_pd['genres'].str.split('|').explode()
    genre_counts = genres.value_counts().head(10)
    genre_counts.plot(kind='bar')
    plt.title('Top 10 Movie Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('movielens_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Execute the pipeline
if __name__ == "__main__":
    main()