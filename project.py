import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")


def predict_ratings(train_matrix):
    means = train_matrix.mean(axis=1)
    normalized = train_matrix.sub(means, axis=0)
    normalized = normalized.fillna(0)

    similarity = cosine_similarity(normalized)
    weighted_sum = similarity.dot(normalized)
    sum_of_weights = np.abs(similarity).sum(axis=1)
    
    predictions = pd.DataFrame(np.divide(weighted_sum.T, sum_of_weights, where=sum_of_weights != 0).T, index=train_matrix.index, columns=train_matrix.columns)    
    predictions = predictions.fillna(0)
    predictions = predictions.add(means, axis=0)
    return predictions

def calculate_MAE_RMSE(predicted, actual):
    actual_flat = actual.stack()
    predicted_flat = predicted.stack()
    common_index = actual_flat.index.intersection(predicted_flat.index)
    actual_common = actual_flat[common_index]
    predicted_common = predicted_flat[common_index]
    mae = mean_absolute_error(actual_common, predicted_common)
    rmse = sqrt(mean_squared_error(actual_common, predicted_common))
    return mae, rmse

def recommend_top_10(predicted, train_matrix):
    recommendations = {}
    for user in predicted.index: 
        if user not in train_matrix.index:
            continue
        already_rated = train_matrix.loc[user].dropna().index.tolist()
        user_predictions = predicted.loc[user]
        user_recommendations = user_predictions[~user_predictions.index.isin(already_rated)]
        top_items = user_recommendations.sort_values(ascending=False).head(10).index.tolist()
        recommendations[user] = top_items
    return recommendations

def evaluate_recommendations(predicted, test_matrix, recommendations):
    precision_list = []
    recall_list = []
    ndcg_list = []

    for user, recommended_items in recommendations.items():
        if user not in test_matrix.index:
            continue

        actual_items = test_matrix.loc[user].dropna().index.tolist()

        relevant_items = set(actual_items).intersection(set(recommended_items))

        precision = len(relevant_items) / 10
        precision_list.append(precision)

        recall = len(relevant_items) / len(actual_items)
        recall_list.append(recall)

        dcg = sum([1 / np.log2(idx + 2) for idx, item in enumerate(recommended_items) if item in relevant_items])
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(actual_items), 10))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)

    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f_measure = (2 * precision * recall / (precision + recall))
    ndcg = np.mean(ndcg_list)

    return precision, recall, f_measure, ndcg

training, testing = train_test_split(ratings, test_size=0.2)
train_matrix = training.pivot(index='userId', columns='movieId', values='rating')
test_matrix = testing.pivot(index='userId', columns='movieId', values='rating')


predicted_ratings = predict_ratings(train_matrix)
mae, rmse = calculate_MAE_RMSE(predicted_ratings, test_matrix)
print(f"MAE: {mae}, RMSE: {rmse}")

top_10_recommendations = recommend_top_10(predicted_ratings, train_matrix)

precision, recall, f_measure, ndcg = evaluate_recommendations(predicted_ratings, test_matrix, top_10_recommendations)
print(f"Precision: {precision}, Recall: {recall}, F-measure: {f_measure}, NDCG: {ndcg}")
