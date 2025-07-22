import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import random


def randomize_interaction_matrix(users_num, items_num):
    all_items = np.array(list(range(items_num)))
    all_users_interactions = []
    unique_users_num = int(users_num/4)

    # randomizing unique users (log of users_num)
    for user in range(unique_users_num):
        interactions_num = np.random.randint(2, items_num/4)
        user_interaction = np.sort(np.random.choice(all_items, size=interactions_num, replace=False))
        all_users_interactions.append(user_interaction)

    # duplicating unique users
    for user in range(users_num-unique_users_num):
        all_users_interactions.append(all_users_interactions[np.random.randint(0, unique_users_num)])

    all_users_interactions, removed_interactions = remove_interaction_for_user(all_users_interactions)

    data_dict = {
        'userId': [idx for idx, _ in enumerate(all_users_interactions) for _ in _],
        'movieId': [val for sublist in all_users_interactions for val in sublist],
        'rating': 1
    }
    df = pd.DataFrame(data_dict)

    return all_users_interactions, removed_interactions, df

def remove_interaction_for_user(all_users_interactions):
    removed_interactions = []
    for i, user_interactions in enumerate(all_users_interactions):
        removed_interaction = np.random.choice(user_interactions, size=1, replace=False)[0]
        all_users_interactions[i] = np.delete(user_interactions, np.where(user_interactions == removed_interaction))
        removed_interactions.append(removed_interaction)
    return all_users_interactions, np.array(removed_interactions)

def convert_all_users_interactions_to_binary_matrix(all_users_interactions, users_num, items_num):
    # Initialize a binary matrix with zeros
    binary_matrix = np.zeros((users_num, items_num), dtype=int)
    # Set 1 at the specified indices
    for row, indices in enumerate(all_users_interactions):
        binary_matrix[row, indices] = 1
    return binary_matrix



def convert_df_to_matrix(users_num, items_num, df):
    # Create a pivot table with specified columns
    pivot_table = df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

    # Reindex the pivot table to ensure all columns are present
    all_columns = np.arange(0, items_num)  # Assuming movie IDs range from 101 to 107
    pivot_table = pivot_table.reindex(columns=all_columns, fill_value=0)

    # Convert the pivot table to a NumPy matrix
    matrix = pivot_table.values

    return matrix


def filter_top_k_users_or_items(df, K, col_to_filter):
    # Filter rows with rating 1
    positive_ratings = df[df['rating'] == 1]
    # Count the number of positive ratings for each user
    counts = positive_ratings[col_to_filter].value_counts()
    # Select the top K
    top_k = counts.head(K).index
    # Filter the original DataFrame based on the top K users
    result_df = df[df[col_to_filter].isin(top_k)]
    return result_df


def filter_items_by_inters_num(df, min_inters_num, max_inters_num):
    movies = list(set(df.movieId.tolist()))
    count = 0
    for movie in movies:
        if (len(df.loc[(df["movieId"] == movie)]) < min_inters_num) or \
                (len(df.loc[(df["movieId"] == movie)]) > max_inters_num):
            count += 1
            df.drop(df.loc[df["movieId"] == movie].index.tolist(), inplace=True)
    print("dropped", count, "items")


def filter_users_by_inters_num(df, min_inters_num, max_inters_num):
    users = list(set(df.userId.tolist()))
    count = 0
    for user in users:
        interactions_count = len(df.loc[(df["userId"] == user) & (df["rating"] == 1)])
        if (interactions_count < min_inters_num) or (interactions_count > max_inters_num):
            count += 1
            df.drop(df.loc[df["userId"] == user].index.tolist(), inplace=True)
    print("dropped", count, "users")


def filter_df_by_ids(df, col_name, ids_range):
    df = df[df[col_name].isin(ids_range)]

    rating_count = df[df['rating'] == 1].groupby('userId')['rating'].count()
    filtered_user_ids = rating_count[rating_count >= 2].index
    df = df[df['userId'].isin(filtered_user_ids)]
    return df

def load_jester_data(users_num, items_num):
    df = pd.read_excel('./data/jester_data/jester-data-3.xls')
    data = df.iloc[:, 1:101]  # Adjust the column range if needed
    data = data.sample(frac=1).reset_index(drop=True)

    matrix = data.to_numpy()
    matrix = np.where(matrix == 99, 0, 1)
    matrix = matrix[:users_num, :items_num]

    all_users_interactions = []
    removed_interactions = []
    df_rows = []

    for user_id, row in enumerate(matrix):
        # Find the indices of all "1"s in the row
        ones_indices = np.where(row == 1)[0].tolist()
        if ones_indices:
            removed_index = random.choice(ones_indices)
            row[removed_index] = 0
            removed_interactions.append(removed_index)
        else:
            removed_interactions.append(None)

        # Update the list of indices of "1"s after removal
        updated_ones_indices = np.where(row == 1)[0].tolist()
        all_users_interactions.append(updated_ones_indices)

        # Append rows to the DataFrame list
        for movie_id in updated_ones_indices:
            df_rows.append({'userId': user_id, 'movieId': movie_id, 'rating': 1})

    # Create the final DataFrame
    df = pd.DataFrame(df_rows, columns=['userId', 'movieId', 'rating'])

    return all_users_interactions, removed_interactions, df


def load_movielens_data(users_num, items_num):
    with open('data/ml-100k/ml-100k/u.data') as f:
        lines = f.readlines()
        lines = [[eval(a) for a in line.split()] for line in lines]
        df = pd.DataFrame(lines, columns=['userId', 'movieId', 'rating', 'timestamp'])

        unique_item_ids = df['movieId'].unique()
        total_unique_items = len(unique_item_ids)

        # Step 2: Count how many unique item IDs each user has interacted with
        user_item_counts = df.groupby('userId')['movieId'].nunique()
        min_threshold = 20
        max_threshold = 80
        users_to_remove = user_item_counts[
            (user_item_counts < min_threshold) | (user_item_counts > max_threshold)].index
        df = df[~df['userId'].isin(users_to_remove)]
        # print("removing", len(users_to_remove))


        rating_threshold = 0
        df.loc[df["rating"] <= rating_threshold, "rating"] = -1
        df.loc[df["rating"] > rating_threshold, "rating"] = 1

        random_int = np.random.randint(0, 500)

        df = filter_df_by_ids(df, 'movieId', range(random_int, random_int+items_num))

        if len(df['userId'].unique()) > users_num:
            df = df[df['userId'].isin(np.random.choice(df['userId'].unique(), size=users_num, replace=False))]

        # ------------ reindexing ------------

        df['userId'] = df.groupby('userId').ngroup()
        df['movieId'] = df.groupby('movieId').ngroup()
        df['timestamp'] = df.groupby('timestamp').ngroup()
        df = df.sort_values(by=['userId', 'movieId', 'timestamp'])
        df = df.reset_index(drop=True)

        # ----------- removing last interaction -----------
        all_users_interactions = []
        removed_interactions = []
        for user_id in df['userId'].unique():
            latest_interaction_index = df.loc[(df['userId'] == user_id) & (df['rating'] == 1)]['timestamp'].idxmax()
            removed_interactions.append(df.loc[latest_interaction_index]['movieId'])
            df = df.drop(latest_interaction_index)
            user_movies = df.loc[df['userId'] == user_id, 'movieId'].tolist()
            # user_movies.remove(removed_interactions[-1])
            all_users_interactions.append(user_movies)
        print(f"users num: {len(df['userId'].unique())} items num: {len(df['movieId'].unique())}")
        return all_users_interactions, removed_interactions, len(all_users_interactions), df
