import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

#Rating prediction df
def predict_ratings(train_matrix):
    means = train_matrix.mean(axis=1)
    normalized = train_matrix.sub(means, axis=0)
    normalized = normalized.fillna(0)

    #similarity = normalized.T.corr(method='jaccard').fillna(0)
    similarity = cosine_similarity(normalized)
    #print(similarity)

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

# MAE and RMSE
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



# recommend 10 ratings based on predicted ratings and the training data matrix

def recommend_top_n(predicted, train_matrix, top_n=10):

    recommendations = {}

    for user in predicted.index:  # Loop through each user
        if user not in train_matrix.index:
            continue  # Skip users without training data

                # Get all movies the user has rated from the original ratings DataFrame
        already_rated = ratings[ratings['userId'] == user]['movieId'].unique()
        
        # Get predicted ratings for the user
        user_predictions = predicted.loc[user]
        
        # Filter out movies the user has already rated
        user_recommendations = user_predictions[~user_predictions.index.isin(already_rated)]
        
        # Sort by predicted ratings and select top-N items
        top_items = user_recommendations.sort_values(ascending=False).head(top_n).index.tolist()
        
        recommendations[user] = top_items
        
    return recommendations

# Need to look at recommendations and test data to get 4 values: precision, recall, f measure, and ndcg

def evaluate_recommendations():
    
    return precision, recall, f_measuer, ndcg


# pre processing data, 80%,20% split
training, testing = train_test_split(ratings, test_size=0.2, random_state=42)

train_matrix = training.pivot(index='userId', columns='movieId', values='rating')
test_matrix = testing.pivot(index='userId', columns='movieId', values='rating')

predicted_ratings = predict_ratings(train_matrix)



#print(test_matrix)
mae, rmse = calculate_MAE_RMSE(predicted_ratings, test_matrix)
print(f"MAE is {mae}, RMSE is {rmse}")

# Generate recommendations
top_n_recommendations = recommend_top_n(predicted_ratings, train_matrix, top_n=10)



# Display recommendations for a specific user
user_id = 1  # Replace with an actual user ID from your dataset
if user_id in top_n_recommendations:
    print(f"Top-10 recommendations for user {user_id}: {top_n_recommendations[user_id]}")
    recommended_movies = movies[movies['movieId'].isin(top_n_recommendations[user_id])]
    print(recommended_movies[['movieId', 'genres']])
    


else:
    print(f"No recommendations for user {user_id}.")



