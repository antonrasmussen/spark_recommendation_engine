#!/usr/bin/env python3
# spark_re.py  ─ MovieLens-32M recommender

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# ------------------------------------------------------------------
DATA_DIR = "data/ml-32m"                    # ← unzip ml-32m.zip here
N_RECS    = 10                              # top-N per user/movie
MEM_GB    = "4g"                            # bump if you have RAM

# 1. Spark session --------------------------------------------------
spark = (SparkSession.builder
         .appName("MovieLens32M_ALS")
         .config("spark.executor.memory", MEM_GB)
         .getOrCreate())

# 2. Load data ------------------------------------------------------
ratings = (spark.read
           .option("header", "true")
           .option("inferSchema", "true")   # lets Spark set data types
           .csv(f"{DATA_DIR}/ratings.csv")
           .select("userId", "movieId", "rating"))

movies  = (spark.read
           .option("header", "true")
           .option("inferSchema", "true")
           .csv(f"{DATA_DIR}/movies.csv"))

tags    = (spark.read
           .option("header", "true")
           .option("inferSchema", "true")
           .csv(f"{DATA_DIR}/tags.csv"))

print(f"⚡  Loaded: {ratings.count():,} ratings | {movies.count():,} movies")

# 3. Train / validation split --------------------------------------
train, valid = ratings.randomSplit([0.8, 0.2], seed=42)

# 4. ALS model & tuning --------------------------------------------
als = (ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
           nonnegative=True, implicitPrefs=False, coldStartStrategy="drop",
           maxIter=15))

param_grid = (ParamGridBuilder()
              .addGrid(als.rank,     [64, 128])
              .addGrid(als.regParam, [0.05, 0.10])
              .build())

tvs = (TrainValidationSplit(estimator=als,
                            estimatorParamMaps=param_grid,
                            evaluator=RegressionEvaluator(metricName="rmse",
                                                          labelCol="rating",
                                                          predictionCol="prediction"),
                            trainRatio=0.9, seed=42))

model = tvs.fit(train).bestModel
rmse  = RegressionEvaluator(metricName="rmse",
                            labelCol="rating",
                            predictionCol="prediction").evaluate(model.transform(valid))
print(f"✔  Best rank={model.rank}  |  Validation RMSE={rmse:.4f}")

# 5. Generate recommendations --------------------------------------
user_recs  = model.recommendForAllUsers(N_RECS)
movie_recs = model.recommendForAllItems(N_RECS)

# 6. Optional: enrich movie-side recs with tag lists ---------------
tag_features = (tags.groupBy("movieId")
                     .agg(F.collect_list("tag").alias("tags")))

movie_recs_enriched = movie_recs.join(tag_features, "movieId", "left")

# 7. (demo) pretty-print top N for one user ------------------------
def show_user_recs(user_id: int, n=N_RECS):
    (user_recs
     .filter(F.col("userId") == user_id)
     .selectExpr("explode(recommendations) AS rec")
     .selectExpr("rec.movieId", "rec.rating AS score")
     .join(movies, "movieId")
     .orderBy(F.col("score").desc())
     .show(n, truncate=False))

show_user_recs(123)

# 8. Persist artefacts ---------------------------------------------
model.save("models/als_movielens32m")
user_recs.write.mode("overwrite").parquet("output/user_topN/")
movie_recs_enriched.write.mode("overwrite").parquet("output/movie_topN/")

spark.stop()
