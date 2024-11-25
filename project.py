import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Predict ratings using collaborative filtering
def predict_ratings(train_matrix):
    means = train_matrix.mean(axis=1)
    normalized = train_matrix.sub(means, axis=0)
    normalized = normalized.fillna(0)

    similarity = cosine_similarity(normalized)
    weighted_sum = similarity.dot(normalized)
    sum_of_weights = np.abs(similarity).sum(axis=1)
    
    predictions = pd.DataFrame(
        np.divide(weighted_sum.T, sum_of_weights, where=sum_of_weights != 0).T,
        index=train_matrix.index,   # Set userId as row index
        columns=train_matrix.columns  # Set movieId as column index
    )    
    predictions = predictions.fillna(0)
    predictions = predictions.add(means, axis=0)
    return predictions

# Calculate MAE and RMSE
def calculate_MAE_RMSE(predicted, actual):
    actual_flat = actual.stack()
    predicted_flat = predicted.stack()
    # Align indices to ensure non-empty arrays
    common_index = actual_flat.index.intersection(predicted_flat.index)
    if len(common_index) == 0:
        raise ValueError("No common indices between predicted and actual ratings.")
    actual_common = actual_flat[common_index]
    predicted_common = predicted_flat[common_index]
    mae = mean_absolute_error(actual_common, predicted_common)
    rmse = sqrt(mean_squared_error(actual_common, predicted_common))
    return mae, rmse

# Recommend top N items for each user
def recommend_top_n(predicted, train_matrix, top_n=10):
    recommendations = {}
    for user in predicted.index:  # Loop through each user
        if user not in train_matrix.index:
            continue  # Skip users without training data
        already_rated = train_matrix.loc[user].dropna().index.tolist()
        user_predictions = predicted.loc[user]
        user_recommendations = user_predictions[~user_predictions.index.isin(already_rated)]
        top_items = user_recommendations.sort_values(ascending=False).head(top_n).index.tolist()
        recommendations[user] = top_items
    return recommendations

# Evaluate recommendations with debugging
def evaluate_recommendations_debug(predicted, test_matrix, recommendations, top_n=10):
    precision_list = []
    recall_list = []
    ndcg_list = []

    for user, recommended_items in recommendations.items():
        if user not in test_matrix.index:
            continue

        # Items in the test data that the user has rated
        actual_items = test_matrix.loc[user].dropna().index.tolist()

        if not actual_items:
            print(f"User {user} has no items in test set.")
            continue

        # Items in the top-N recommendations that are also in the actual test data
        relevant_items = set(actual_items).intersection(set(recommended_items))

        print(f"User {user}:\n  Recommended: {recommended_items}\n  Actual: {actual_items}\n  Relevant: {relevant_items}")

        # Precision: |Relevant items ∩ Recommended items| / |Recommended items|
        precision = len(relevant_items) / top_n if top_n > 0 else 0
        precision_list.append(precision)

        # Recall: |Relevant items ∩ Recommended items| / |Relevant items|
        recall = len(relevant_items) / len(actual_items) if len(actual_items) > 0 else 0
        recall_list.append(recall)

        # NDCG: Normalized Discounted Cumulative Gain
        dcg = sum([1 / np.log2(idx + 2) for idx, item in enumerate(recommended_items) if item in relevant_items])
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(actual_items), top_n))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)

    # Handle case where no metrics are computed
    if not precision_list or not recall_list or not ndcg_list:
        print("No valid users for metrics computation.")
        return 0, 0, 0, 0

    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f_measure = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    ndcg = np.mean(ndcg_list)

    return precision, recall, f_measure, ndcg

# Preprocessing data: 80% training, 20% testing
training, testing = train_test_split(ratings, test_size=0.2, random_state=42)
train_matrix = training.pivot(index='userId', columns='movieId', values='rating')
test_matrix = testing.pivot(index='userId', columns='movieId', values='rating')

# Predict ratings and calculate metrics
predicted_ratings = predict_ratings(train_matrix)
mae, rmse = calculate_MAE_RMSE(predicted_ratings, test_matrix)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Generate top-N recommendations
top_n_recommendations = recommend_top_n(predicted_ratings, train_matrix, top_n=10)

# Evaluate recommendations
precision, recall, f_measure, ndcg = evaluate_recommendations_debug(
    predicted_ratings, test_matrix, top_n_recommendations, top_n=10
)

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-measure: {f_measure:.4f}, NDCG: {ndcg:.4f}")

# Display recommendations for a specific user
user_id = 1  # Replace with an actual user ID from your dataset
if user_id in top_n_recommendations:
    print(f"Top-10 recommendations for user {user_id}: {top_n_recommendations[user_id]}")
    recommended_movies = movies[movies['movieId'].isin(top_n_recommendations[user_id])]
    print(recommended_movies[['movieId', 'genres']])
else:
    print(f"No recommendations for user {user_id}.")
