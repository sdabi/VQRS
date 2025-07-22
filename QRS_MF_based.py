
from multiprocessing import Pool
from utils_func import *
import pennylane as qml
from pennylane import numpy as np
import math
from pennylane.templates.layers import StronglyEntanglingLayers

PI = 3.14
PI_2 = PI / 2
PI_4 = PI_2 / 2

PROPS_SUM_INTERACTED_ITEMS = 1
QUBITS_NUM = 6

def print_tensor(STR, probs, green_list, blue_item=""):
    str_to_print = STR
    max_index_search_arr = qml.math.toarray(probs).copy()
    max_index_search_arr[green_list!=0] = -1
    max_index = np.argmax(max_index_search_arr)
    del max_index_search_arr
    for i,item in enumerate(qml.math.toarray(probs)):
        if i == blue_item:
            str_to_print += colored(BLUE, '{val:>5} '.format(val=round(item, 3)))
        elif i == max_index:
            str_to_print += colored(ORANGE, '{val:>5} '.format(val=round(item, 3)))
        elif green_list[i]!=0:
            str_to_print += colored(GREEN, '{val:>5} '.format(val=round(item, 3)))
        else:
            str_to_print += '{val:>5} '.format(val=round(item, 3))
    print(str_to_print)



dev_user_params_circ = qml.device('default.qubit', wires=QUBITS_NUM)
@qml.qnode(dev_user_params_circ)
def user_params_circ(user_params):
    for wire in range(QUBITS_NUM):
        qml.Hadamard(wire)
    for q in range(QUBITS_NUM):
        qml.RY(user_params[q], wires=q)
    return qml.state()



qnode_probs_no_ent = qml.device('default.qubit', wires=QUBITS_NUM)
@qml.qnode(qnode_probs_no_ent)
def QRS_circ_probs_no_ent(params, user_params):
    for wire in range(QUBITS_NUM):
        qml.Hadamard(wire)

    for layer in params:
        for q in range(QUBITS_NUM):
            qml.RY(user_params[q], wires=q)
            qml.RY(layer[q], wires=q)
    return qml.probs(wires=range(QUBITS_NUM))



qnode_probs = qml.device('default.qubit', wires=QUBITS_NUM)
@qml.qnode(qnode_probs)
def QRS_circ_probs(params, user_params):
    for wire in range(QUBITS_NUM):
        qml.Hadamard(wire)

    for layer in params:
        for q in range(QUBITS_NUM):
            qml.RY(user_params[q], wires=q)
            qml.RY(layer[q], wires=q)
            qml.CNOT(wires=[q, ((q+1)%QUBITS_NUM)])
    return qml.probs(wires=range(QUBITS_NUM))



qnode_samples = qml.device('default.qubit', wires=QUBITS_NUM, shots=1)
@qml.qnode(qnode_samples)
def QRS_circ_samples(params, user_params):
    for wire in range(QUBITS_NUM):
        qml.Hadamard(wire)

    for layer in params:
        for q in range(QUBITS_NUM):
            qml.RY(user_params[q], wires=q)
            qml.RY(layer[q], wires=q)
            qml.CNOT(wires=[q, ((q+1)%QUBITS_NUM)])
    return qml.sample(wires=range(QUBITS_NUM))







class QRS_MF():
    def __init__(self, users_nums, items_num, all_users_interactions, removed_interactions, layers, users_vectors=""):
        self.perf_col = performace_collector()
        self.users_nums, self.items_num = users_nums, items_num
        self.all_users_interactions = all_users_interactions
        self.removed_interactions = removed_interactions

        self.target_probs = []
        for user in range(users_nums):
            self.target_probs.append(self.get_target_probs(user))

        # self.user_vecs_norms = calculate_norms(users_vectors)

        self.params = np.zeros((layers, QUBITS_NUM), requires_grad=True)

        self.prev_cost = 100

        # multiply_factor = PI_4/ np.max(np.abs(users_vectors))
        multiply_factor = 1
        self.user_params = []
        for user in range(users_nums):
            x = np.array(users_vectors[user], requires_grad=True) * multiply_factor
            self.user_params.append(x)

        self.samples_count = "inf"
        self.get_cost_for_epoch()
        self.optimizer = qml.AdagradOptimizer(stepsize=0.1, eps=1e-08)
        # self.user_params = np.load("user_params.npy")
        # self.optimize_users_params()

        # ~ ~ ~ lines for debug PCA~ ~ ~
        # user_states_0, user_states_1, user_states_PI_16, user_states_PI_8, user_states_PI_4, user_states_PI_2, user_states_PI, user_states_2PI, user_states_4PI= [], [], [], [], [], [], [], [], []
        #
        # print(np.max(users_vectors), np.min(users_vectors))
        #
        # multiply_factor_4PI = 4*PI / np.max(np.abs(users_vectors))
        # multiply_factor_2PI = multiply_factor_4PI / 2
        # multiply_factor_PI = multiply_factor_2PI / 2
        # multiply_factor_PI_2 = multiply_factor_PI / 2
        # multiply_factor_PI_4 = multiply_factor_PI_2 / 2
        # multiply_factor_PI_8 = multiply_factor_PI_4 / 2
        # multiply_factor_PI_16 = multiply_factor_PI_8 / 2
        #
        # for user in range(users_nums):
        #     if user ==0:
        #         print(users_vectors[user])
        #     user_states_4PI.append(
        #         user_params_circ(np.array(users_vectors[user], requires_grad=True) * multiply_factor_4PI))
        #     user_states_2PI.append(
        #         user_params_circ(np.array(users_vectors[user], requires_grad=True) * multiply_factor_2PI))
        #     user_states_PI.append(
        #         user_params_circ(np.array(users_vectors[user], requires_grad=True) * multiply_factor_PI))
        #     user_states_PI_2.append(
        #         user_params_circ(np.array(users_vectors[user], requires_grad=True) * multiply_factor_PI_2))
        #     user_states_PI_4.append(
        #         user_params_circ(np.array(users_vectors[user], requires_grad=True) * multiply_factor_PI_4))
        #     user_states_PI_8.append(
        #         user_params_circ(np.array(users_vectors[user], requires_grad=True) * multiply_factor_PI_8))
        #     user_states_PI_16.append(
        #         user_params_circ(np.array(users_vectors[user], requires_grad=True) * multiply_factor_PI_16))
        #     user_states_1.append(
        #         user_params_circ(np.array(users_vectors[user], requires_grad=True) * 1))
        #
        # apply_pca_and_plot([users_vectors,  np.abs(user_states_1)**2],
        #                    ["P Matrix Vectors", "Users' probabilities vectors"])
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

        # np.save("user_params.npy", self.user_params)

    def load_params(self, params):
        self.params = params
    def get_target_probs(self, user):
        target = np.zeros(self.items_num)
        target[self.all_users_interactions[user]] = PROPS_SUM_INTERACTED_ITEMS / len(self.all_users_interactions[user])
        return target

    # ===================================== Optimizing Global Params =====================================

    def train(self, step_size, epochs):
        self.params.requires_grad = True

        for epoch in range(epochs):
            print("epoch", epoch)
            self.params = self.optimizer.step(self.optimize_global_params_all_at_once, self.params)
            cost = self.get_cost_for_epoch()


    def optimize_global_params_all_at_once(self, params):
        overall_err = 0
        for user in range(self.users_nums):
            target = self.target_probs[user]
            probs = QRS_circ_probs(params, self.user_params[user])
            for inter in self.all_users_interactions[user]:
                overall_err += ((probs[inter] - target[inter])**2)
        return overall_err


    def get_cost_for_epoch(self):
        cost = 0
        for user in range(self.users_nums):
            target = self.target_probs[user]
            probs = QRS_circ_probs(self.params, self.user_params[user])
            for inter in self.all_users_interactions[user]:
                cost += ((probs[inter] - target[inter])**2)
            self.perf_col.add_epoch_data(probs.numpy(), self.all_users_interactions[user], self.removed_interactions[user])
            # print_tensor(f'User {user} Prob Vec: ', probs, target, self.removed_interactions[user])
        # print(f'cost: {cost} - {self.prev_cost-cost}')
        self.prev_cost = cost
        self.perf_col.finalize_epoch_data(cost.item())
        return cost


    def get_recommendation(self, user, items):
        probs = 0
        # print(user, self.samples_count)
        # print("reco for user:", user, "samples:", self.samples_count)
        if self.samples_count == "inf":
            probs = QRS_circ_probs(self.params, self.user_params[user])
        else:
            if self.samples_count == "items_num": shots = self.items_num
            elif self.samples_count == "sqrt"   : shots = int(math.ceil(np.sqrt(self.items_num)))
            elif self.samples_count == "log2"   : shots = int(np.log2(self.items_num))
            decimals = []
            valid_shots = 0
            total_shots = 0
            while valid_shots < shots:
                total_shots += 1
                sample = QRS_circ_samples(self.params, self.user_params[user])
                decimal = int(''.join(map(str, sample)), 2)
                if decimal not in self.all_users_interactions[user]:
                    valid_shots += 1
                    decimals.append(decimal)

            probs = np.bincount(decimals, minlength=self.items_num)
            probs = probs / np.sum(probs)
            print(user, total_shots, valid_shots)

        # print_tensor(" PROBS ", probs, self.target_probs[user], self.removed_interactions[user])
        return probs[items]

    def get_probs_for_user(self, user):
        if self.samples_count != "inf": exit("error - get_probs_for_user only valid with inf mode")
        return QRS_circ_probs(self.params, self.user_params[user]).numpy()

    def train_no_ent(self, step_size, epochs):
        self.params.requires_grad = True

        for epoch in range(epochs):
            print("epoch", epoch)
            self.params = self.optimizer.step(self.optimize_global_params_all_at_once_no_ent, self.params)


    def optimize_global_params_all_at_once_no_ent(self, params):
        overall_err = 0

        for user in range(self.users_nums):
            target = self.target_probs[user]
            probs = QRS_circ_probs_no_ent(params, self.user_params[user])
            for inter in self.all_users_interactions[user]:
                overall_err += ((probs[inter] - target[inter])**2)
        return overall_err


class performace_collector():
    def __init__(self):
        self.avg_sum_greens = []
        self.avg_sum_greys = []
        self.avg_blue  = []
        self.avg_green = []
        self.avg_grey = []
        self.cost = []

        self.current_epoch_greens = []
        self.current_epoch_blues = []
        self.current_epoch_greys = []

    def add_epoch_data(self, probs_vec, green_items, blue_item):
        self.current_epoch_greens.append(probs_vec[green_items])
        self.current_epoch_blues.append(probs_vec[blue_item])
        self.current_epoch_greys.append([prob for i, prob in enumerate(probs_vec) if i not in green_items and i != blue_item])

    def finalize_epoch_data(self, cost):

        avg_green, avg_sums_greens = self.get_sum_avg_and_avg_val(self.current_epoch_greens)

        self.avg_green.append(avg_green)
        self.avg_sum_greens.append(avg_sums_greens)

        avg_grey, avg_sums_greys = self.get_sum_avg_and_avg_val(self.current_epoch_greys)
        self.avg_grey.append(avg_grey)
        self.avg_sum_greys.append(avg_sums_greys)

        self.avg_blue.append(sum(self.current_epoch_blues) / len(self.current_epoch_blues))

        self.cost.append(cost)

        self.current_epoch_greens = []
        self.current_epoch_blues = []
        self.current_epoch_greys = []

    def get_sum_avg_and_avg_val(self, list_of_lists):
        total_sum = 0
        total_elements = 0
        sum_of_inner_lists = []

        for inner_list in list_of_lists:
            inner_sum = sum(inner_list)
            sum_of_inner_lists.append(inner_sum)
            total_sum += inner_sum
            total_elements += len(inner_list)
        return total_sum / total_elements, sum(sum_of_inner_lists) / len(sum_of_inner_lists)


    def plot_avgs(self, save_plot_dir=""):
        # ------------- plot avgs sum ---------------
        x = range(len(self.avg_sum_greens))
        plt.stackplot(x, self.avg_sum_greens, self.avg_blue, self.avg_sum_greys,
                      labels=['Green items', 'Blue item', 'Gray items'],
                      colors=['#59A558', '#2C66B9', '#BCBCBC'])

        plt.xlabel('Epochs')
        plt.ylabel('AVG Amplitude')
        plt.title('AVG Prob Sum')
        plt.xticks(range(0, len(self.cost), 5))
        plt.legend(loc='upper right')

        if save_plot_dir != "": plt.savefig(save_plot_dir+'avgs_of_sums')  # Change the path and filename as needed
        plt.show()

        # ------------- plot items avgs ---------------
        fig, cost_ax = plt.subplots(figsize=(8, 8))

        # Plot the cost values on the left y-axis
        x = range(len(self.cost))
        cost_ax.plot(x, self.cost, 'black', label='Cost', linestyle=':', linewidth=7)
        cost_ax.set_xlabel('Epochs',fontsize=22)
        cost_ax.set_ylabel('Cost',fontsize=22)
        cost_ax.tick_params(axis='y', labelsize=22)

        # Create a second y-axis
        avgs_ax = cost_ax.twinx()

        # Plot the averages on the right y-axis
        avgs_ax.plot(x, self.avg_green, '#2C66B9', label=r'$\overline{p}(\text{interacted items})$', linewidth=7)
        avgs_ax.plot(x, self.avg_blue,  '#59A558', label=r'$\overline{p}(\text{true item})$', linewidth=7)
        avgs_ax.plot(x, self.avg_grey,  'brown', label=r'$\overline{p}(\text{non-interacted items})$', linewidth=7)
        avgs_ax.set_ylabel('AVG probs', color='k',fontsize=22)
        avgs_ax.tick_params(axis='y', labelcolor='k', labelsize=22)

        plt.xticks(range(0, len(self.cost), 5) ,fontsize=22)
        plt.grid(True, linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        handles, labels = plt.gca().get_legend_handles_labels()

        # fig.legend(fontsize=19, bbox_to_anchor=(0.842, 0.9035)) #5 qubit
        # fig.legend(fontsize=19, bbox_to_anchor=(0.835, 0.6035)) #6 qubit
        fig.legend(fontsize=19, bbox_to_anchor=(0.8, 0.75)) #7 qubit
        if save_plot_dir != "": plt.savefig(save_plot_dir+'cost_and_avgs')  # Change the path and filename as needed
        plt.show()


    def print_class_variables(self):
        class_vars = {k: v for k, v in self.__dict__.items() if not callable(v) and not k.startswith('__')}
        for name, value in class_vars.items():
            print(f"{name}: {value}")

    def export_data(self):
        return np.array([self.cost, self.avg_sum_greens, self.avg_sum_greys, self.avg_blue, self.avg_green, self.avg_grey])

    def import_data(self, import_data):
        self.cost = import_data[0]
        self.avg_sum_greens = import_data[1]
        self.avg_sum_greys = import_data[2]
        self.avg_blue = import_data[3]
        self.avg_green = import_data[4]
        self.avg_grey = import_data[5]
