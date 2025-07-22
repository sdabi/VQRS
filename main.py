
import math
import numpy as np
from QRS_MF_based import QRS_MF
from MF2 import matrix_factorization2
from history_removal import HistRem
from popularity_RS import popularity_RS
from pytourch_MF import *
from random_RS import random_RS
from data_loading import load_jester_data, convert_df_to_matrix, load_movielens_data
from utils_func import *
from multiprocessing import Pool
from jester_test import *
import random

INTERESTING_PLOT = True
NOT_INTERESTING_PLOT = False

def test_model(model, all_users_interactions, removed_interactions, items_num):
    HR_till_10 = np.zeros(11)
    for user, (users_interactions, removed_interaction) in enumerate(zip(all_users_interactions, removed_interactions)):
        user_non_interacted = np.array([item for item in list(range(items_num)) if item not in users_interactions])

        reco_scores_full_array = np.zeros(items_num)
        reco_scores_full_array[user_non_interacted] = model.get_recommendation(user, user_non_interacted)
        recommended_item_score = reco_scores_full_array[removed_interaction]
        if recommended_item_score == 0: continue

        sorted_list_desc = sorted(reco_scores_full_array, reverse=True)
        occurrences = [i for i, x in enumerate(sorted_list_desc) if x == recommended_item_score]
        removed_interaction_pos = random.choice(occurrences) + 1 # 1 based counting

        if removed_interaction_pos < 11: HR_till_10[removed_interaction_pos] += 1

    return (np.cumsum(HR_till_10)/len(all_users_interactions))


def single_run(seed=0):
    print("seed:", seed)
    np.random.seed(seed)

    HRK_lists = []
    HRK_samples_lists = []

    K = math.ceil(math.log(items_num, 2))

    # jester
    users_nums = users_nums_i
    all_users_interactions, removed_interactions, df = load_jester_data(users_nums_i, items_num)
    R = convert_df_to_matrix(users_nums_i, items_num, df)

    # movielens
    # all_users_interactions, removed_interactions, users_nums, df = load_movielens_data(users_nums_i, items_num)
    # R = convert_df_to_matrix(users_nums, items_num, df)

    # ------------- Random recommendation -------------
    HRK_accum = np.zeros(11)
    for i in range(30):
        RAND_RS = random_RS(R)
        HRK = test_model(RAND_RS, all_users_interactions, removed_interactions, items_num)
        HRK_accum += HRK
    HRK_accum /= 30
    HRK_lists.append((HRK_accum, "RAND", INTERESTING_PLOT))

    # ------------- Popularity recommendation -------------
    POP_RS = popularity_RS(R)
    HRK = test_model(POP_RS, all_users_interactions, removed_interactions, items_num)
    HRK_lists.append((HRK, "POP", INTERESTING_PLOT))

    # ------------- Matrix Factorization -----------------
    mf2 = matrix_factorization2(df, K, users_nums, items_num)
    HRK = test_model(mf2, all_users_interactions, removed_interactions, items_num)
    HRK_lists.append((HRK, "MF", INTERESTING_PLOT))
    user_vecs = mf2.get_user_vecs()

    # ------------- QRS -----------------
    QRS = QRS_MF(users_nums, items_num, all_users_interactions, removed_interactions, 6, user_vecs)
    for (step_size, epochs) in [(0.1, 50)]:
        QRS.train(step_size, epochs)

    QRS.perf_col.plot_avgs()

    HRK = test_model(QRS, all_users_interactions, removed_interactions, items_num)
    HRK_lists.append((HRK, "VQRS", INTERESTING_PLOT))

    plot_HRK(HRK_lists, f'Hit Rate K - {users_nums} Users, {items_num} Items')
    plot_HRK_simple(HRK_lists, top_K=5)

    # --------- Test Number of Shots -----------
    for samples in ["inf", "items_num", "sqrt", "log2"]:
        QRS.samples_count = samples
        sets = []
        for i in range(20):
            sets.append((QRS, all_users_interactions, removed_interactions, items_num))
        with Pool(processes=2) as pool:
            results = pool.map(parallel_shots_testing, sets)
        HRK_accum = np.array([sum(x) for x in zip(*results)])
        HRK_accum /= len(sets)
        HRK_samples_lists.append((HRK_accum, "QRS_"+samples, INTERESTING_PLOT))
    plot_HRK(HRK_samples_lists, f'Hit Rate K - Per shots number')
    # plot_HRK_shoots_simple(HRK_samples_lists, top_K=5, save_plot_dir=save_plot_dir+'HRK_shots_simple')
    # plot_HRK_shots_bars(HRK_samples_lists, top_K=5, save_plot_dir=save_plot_dir+'HRK_shots_bars')

    # ---------- History Removal -----------------
    hr_model = HistRem(users_nums, items_num, all_users_interactions, user_vecs, QRS.params)
    hr_model.train(0.1, 1)
    HRK = test_model(hr_model, all_users_interactions, removed_interactions, items_num)
    HRK_lists.append((HRK, "VQRS_HR", INTERESTING_PLOT))
    plot_HRK(HRK_lists, f'Hit Rate K - QRS with Hist Rem')
    plot_HRK_simple(HRK_lists, top_K=5)


def parallel_shots_testing(set):
    (QRS, all_users_interactions, removed_interactions, items_num) = set
    return test_model(QRS, all_users_interactions, removed_interactions, items_num)


def test_entanglement(users_nums, items_num, all_users_interactions, removed_interactions, user_vecs):
    HRK_lists = []
    epochs_and_LR = [(0.1, 10)]

    QRS = QRS_MF(users_nums, items_num, all_users_interactions, removed_interactions, 10, user_vecs)

    for (step_size, epochs) in epochs_and_LR:
        QRS.train_no_ent(step_size, epochs)
        HRK = test_model(QRS, all_users_interactions, removed_interactions, items_num)
        HRK_lists.append((HRK, "QRS_no_ent", INTERESTING_PLOT))
        QRS.train(step_size, epochs)
        HRK = test_model(QRS, all_users_interactions, removed_interactions, items_num)
        HRK_lists.append((HRK, "QRS", INTERESTING_PLOT))

    plot_HRK_simple(HRK_lists, top_K=5)


#GLOBALS:
users_nums_i, items_num = 128,128

if __name__ == '__main__':
    single_run()
