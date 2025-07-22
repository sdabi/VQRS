from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
import numpy as np

from surprise import accuracy
import pandas as pd

class matrix_factorization2():
    def __init__(self, df, D, users_num, items_num):
        reader = Reader(rating_scale=(0, 1))

        all_users = np.arange(users_num)  # users from 0 to users_nums-1
        all_items = np.arange(items_num)  # movies from 0 to items_num-1

        # Create a DataFrame with all possible combinations of users and movies
        user_movie_pairs = pd.MultiIndex.from_product([all_users, all_items], names=['userId', 'movieId']).to_frame(
            index=False)

        # Step 3: Merge the original ratings with the user-movie pairs (this will add NaNs for missing ratings)
        df_full = pd.merge(user_movie_pairs, df[['userId', 'movieId', 'rating']], on=['userId', 'movieId'], how='left')

        # Step 4: Fill the NaN values with 0 for missing ratings
        df_full['rating'] = df_full['rating'].fillna(0)

        dataset = Dataset.load_from_df(df_full[['userId', 'movieId', 'rating']], reader)

        # Step 3: Split the data into training and testing sets
        trainset, testset = train_test_split(dataset, test_size=0.2)

        # Step 4: GridSearch to tune hyperparameters of SVD
        param_grid = {
            'n_factors': [D],
            'n_epochs': [100],
            'reg_all': [0.02, 0.1, 0.2, 0.3, 0.5],
            'lr_all': [0.002, 0.005, 0.01, 0.1],
        }
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(dataset)

        # Best model after grid search
        self.best_model = gs.best_estimator['rmse']

        # Step 5: Train the model with the best parameters
        self.best_model.fit(trainset)


    def get_recommendation(self, user, items):
        scores = []
        for item in items:
            scores.append(self.best_model.predict(user, item).est)
        return scores

    def get_user_vecs(self):
        user_factors = self.best_model.pu  # P matrix
        return np.array(user_factors)


    def print_vecs(self):
        print(np.array(self.best_model.pu))
        print(np.array(self.best_model.qi))
        print(np.dot(self.best_model.pu, self.best_model.qi.T))
        return