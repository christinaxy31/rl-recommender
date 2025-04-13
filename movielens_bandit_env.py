# movielens_bandit_env.py
import numpy as np
import tensorflow_datasets as tfds
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class RealMovieLensEmbeddingEnv(py_environment.PyEnvironment):
    def __init__(self, num_users=100, num_movies=100, embedding_dim=16):
        super().__init__()
        self._num_users = num_users
        self._num_movies = num_movies
        self._embedding_dim = embedding_dim

        raw_ratings = tfds.as_numpy(
            tfds.load('movielens/100k-ratings', split='train', batch_size=-1)
        )

        user_ids = raw_ratings['user_id']
        movie_ids = raw_ratings['movie_id']
        ratings = raw_ratings['user_rating']

        unique_users = np.unique(user_ids)[:num_users]
        unique_movies = np.unique(movie_ids)[:num_movies]
        self._user2idx = {u: i for i, u in enumerate(unique_users)}
        self._movie2idx = {m: i for i, m in enumerate(unique_movies)}

        mask = np.isin(user_ids, unique_users) & np.isin(movie_ids, unique_movies)
        self._user_ids = user_ids[mask]
        self._movie_ids = movie_ids[mask]
        self._ratings = ratings[mask]
        self._num_samples = len(self._ratings)


        #self._user_embeddings = np.random.normal(size=(num_users, embedding_dim)).astype(np.float32)
        #self._movie_embeddings = np.random.normal(size=(num_movies, embedding_dim)).astype(np.float32)

        self._user_embeddings = np.random.normal(0, 1, size=(num_users, embedding_dim)).astype(np.float32)


        self._movie_embeddings = np.array([
            self._user_embeddings[i % num_users] + np.random.normal(0, 0.1, size=embedding_dim)
            for i in range(num_movies)
        ]).astype(np.float32)


        self._context_dim = embedding_dim * 2
        self._index = 0
        self._done = False

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._context_dim,), dtype=np.float32, minimum=-10.0, maximum=10.0
        )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1
        )

    def observation_spec(self): #32 dimensions, [user embedding, movie embedding] vector
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self): #train the same sample [user embedding, movie embedding] multiple times
        self._index = (self._index + 1) % self._num_samples
        self._done = False
        return ts.restart(self._get_context())

    def _step(self, action):
        if self._done:
            return self.reset()

        action = int(action)
        rating = self._ratings[self._index]
        reward = 1.0 if (action == 1 and rating >= 4.0) else 0.0

        self._done = True
        return ts.termination(self._get_context(), reward)

    def _get_context(self):
        user_idx = self._user2idx[self._user_ids[self._index]]
        movie_idx = self._movie2idx[self._movie_ids[self._index]]
        user_vec = self._user_embeddings[user_idx]
        movie_vec = self._movie_embeddings[movie_idx]
        return np.concatenate([user_vec, movie_vec])
