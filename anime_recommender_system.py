



import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Reader, Dataset, SVD



anime_interactions_df = pd.read_csv("/content/rating.csv")

data = Dataset.load_from_df(anime_interactions_df[['user_id', 'anime_id', 'rating']], Reader)

import pandas as pd

file_path = '/content/main_pred.csv'

columns_to_use = ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating_x']

anime_data = pd.read_csv(file_path, usecols=columns_to_use)

anime_data.head()

anime_data['genre'] = anime_data['genre'].str.split(',')

anime_data_expanded = anime_data.explode('genre')

genre_counts = anime_data_expanded['genre'].value_counts()

print(genre_counts)

anime_data['genre'].value_counts()

"""### **Functions**"""

def get_genre():
    # Prompt user for desired genre
    genre = input("Enter your preferred genre: ")
    return genre

def get_type():
    # Provide options for anime type
    type_options = ["Movie", 'TV', 'OVA', 'Special', 'Movie' ,'ONA', 'Music' ]

    # Prompt user for desired type
    while True:
        type_choice = input("Choose your preferred type: ")

        if type_choice in type_options:
            return type_choice
        else:
            print("Invalid choice. Please select a valid option.")

def get_duration():
    # Provide options for anime duration
    duration_options = ["Short", "Medium", "Long"]

    # Prompt user for desired duration
    while True:
        duration_choice = input("Choose your preferred duration (Short/Medium/Long): ")

        if duration_choice in duration_options:
            return duration_choice
        else:
            print("Invalid choice. Please select a valid option.")

user_genre = get_genre()
user_type = get_type()
user_duration = get_duration()


tfidf_vectorizer = TfidfVectorizer()
anime_data["genre_vector"] = list(tfidf_vectorizer.fit_transform(anime_data["genre"].astype(str)).toarray())

user_genre_vector = tfidf_vectorizer.transform([user_genre]).toarray()

filtered_anime = anime_data[anime_data["type"] == user_type]



reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(anime_interactions_df[['user_id', 'anime_id', 'rating']], reader)



model = SVD(n_factors=100)
model.fit(data.build_full_trainset())

cf_predictions = []

for anime_id in filtered_anime["anime_id"]:
    prediction = model.predict(uid=1, iid=anime_id)
    cf_predictions.append(prediction)

genre_similarity = cosine_similarity(user_genre_vector, list(filtered_anime["genre_vector"]))

hybrid_scores = genre_similarity.flatten() + [prediction.est for prediction in cf_predictions]

filtered_anime["hybrid_scores"] = hybrid_scores

"""### **Recommendation**"""

top_k = 5
recommended_anime = filtered_anime.sort_values(by="hybrid_scores", ascending=False).head(top_k)

# Print recommendations
print("Recommended Anime:")
for i in range(len(recommended_anime)):
    print(f"{i+1}. {recommended_anime.iloc[i]['name']}")

