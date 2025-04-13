import numpy as np
import json
import os
import tensorflow_datasets as tfds
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, Sequential

# Set the directory to save the embeddings and mappings
save_dir = 'exported_embeddings'
os.makedirs(save_dir, exist_ok=True)

# Load the MovieLens 100K dataset
raw = tfds.as_numpy(tfds.load('movielens/100k-ratings', split='train', batch_size=-1))
user_ids, movie_ids = raw['user_id'], raw['movie_id']
movie_genres = raw['movie_genres']
user_age = raw['raw_user_age']
user_gender = raw['user_gender']
user_occupation = raw['user_occupation_label']

# Obtain unique user and movie IDs
unique_users = np.unique(user_ids)
unique_movies = np.unique(movie_ids)
user2idx = {u: i for i, u in enumerate(unique_users)}
movie2idx = {m: i for i, m in enumerate(unique_movies)}

# Save the mapping relationships
with open(os.path.join(save_dir, 'user2idx.json'), 'w') as f:
    json.dump({str(k): v for k, v in user2idx.items()}, f)
with open(os.path.join(save_dir, 'movie2idx.json'), 'w') as f:
    json.dump({str(k): v for k, v in movie2idx.items()}, f)

# Process movie embeddings
mlb = MultiLabelBinarizer()
genre_multi_hot = mlb.fit_transform(movie_genres)
genre_embed_net = Sequential([
    layers.InputLayer(input_shape=(genre_multi_hot.shape[1],)),
    layers.Dense(16)
])
movie_embeddings = genre_embed_net(genre_multi_hot).numpy()
np.save(os.path.join(save_dir, 'movie_embeddings.npy'), movie_embeddings)

# Process user embeddings
age_norm = ((user_age - 18) / 50).reshape(-1, 1)
user_gender = np.array(user_gender, dtype='S')
decoded_gender = np.array([g.decode('utf-8') for g in user_gender])
gender_onehot = (decoded_gender == 'M').astype(np.float32).reshape(-1, 1)
num_occupations = np.max(user_occupation) + 1
occupation_onehot = np.eye(num_occupations)[user_occupation]
user_features = np.concatenate([age_norm, gender_onehot, occupation_onehot], axis=1)
user_embed_net = Sequential([
    layers.InputLayer(input_shape=(user_features.shape[1],)),
    layers.Dense(16)
])
user_embeddings = user_embed_net(user_features).numpy()
np.save(os.path.join(save_dir, 'user_embeddings.npy'), user_embeddings)

print(f"Embeddings and mapping relationships have been saved to the directory: {save_dir}")
