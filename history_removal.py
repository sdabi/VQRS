
from multiprocessing import Pool
from utils_func import *
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers

PI = 3.15
PI_2 = PI / 2

QUBITS_NUM = 5

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


qrs_qnode_probs = qml.device('default.qubit', wires=QUBITS_NUM)
@qml.qnode(qrs_qnode_probs)
def QRS_circ_probs(params, user_params):
    for wire in range(QUBITS_NUM):
        qml.Hadamard(wire)

    for layer in params:
        for q in range(QUBITS_NUM):
            qml.RY(user_params[q], wires=q)
            qml.RY(layer[q], wires=q)
            qml.CNOT(wires=[q, ((q+1)%QUBITS_NUM)])
    return qml.probs(wires=range(QUBITS_NUM))




hr_qnode_probs = qml.device('default.qubit', wires=QUBITS_NUM)
@qml.qnode(hr_qnode_probs)
def HR_circ_probs(params, user_params, hr_params):
    for wire in range(QUBITS_NUM):
        qml.Hadamard(wire)

    for layer in params:
        for q in range(QUBITS_NUM):
            qml.RY(user_params[q], wires=q)
            qml.RY(layer[q], wires=q)
            qml.CNOT(wires=[q, ((q+1)%QUBITS_NUM)])
    for layer in hr_params:
        for q in range(QUBITS_NUM):
            qml.RY(layer[q], wires=q)
    return qml.probs(wires=range(QUBITS_NUM))



hr_qnode_samples = qml.device('default.qubit', wires=QUBITS_NUM, shots=1)
@qml.qnode(hr_qnode_samples)
def HR_circ_samples(params, user_params,hr_params):
    for wire in range(QUBITS_NUM):
        qml.Hadamard(wire)

    for layer in params:
        for q in range(QUBITS_NUM):
            qml.RY(user_params[q], wires=q)
            qml.RY(layer[q], wires=q)
            qml.CNOT(wires=[q, ((q+1)%QUBITS_NUM)])
    for layer in hr_params:
        for q in range(QUBITS_NUM):
            qml.RY(layer[q], wires=q)
    return qml.sample(wires=range(QUBITS_NUM))


class HistRem():
    def __init__(self, users_nums, items_num, all_users_interactions, users_vectors, qrs_optimized_params):
        self.users_nums, self.items_num = users_nums, items_num
        self.all_users_interactions = all_users_interactions

        self.qrs_optimized_params = np.array(qrs_optimized_params)
        self.qrs_optimized_params.requires_grad=False

        multiply_factor = PI_2 / np.max(np.abs(users_vectors))
        self.user_params = []
        for user in range(users_nums):
            x = np.array(users_vectors[user], requires_grad=False)
            self.user_params.append(x)

        self.target_probs = []
        for user in range(users_nums):
            self.target_probs.append(self.get_target_probs(user))

        self.params = []
        for user in range(users_nums):
            self.params.append(np.zeros((1,QUBITS_NUM), requires_grad=True))

        self.samples_count = "inf"

    def load_params(self, params):
        self.params = params
    def get_target_probs(self, user):
        qrs_circ_output_probs = QRS_circ_probs(self.qrs_optimized_params, self.user_params[user])
        qrs_circ_output_probs[self.all_users_interactions[user]] = 0
        target = qrs_circ_output_probs / np.sum(qrs_circ_output_probs)
        return target

    # ===================================== Optimizing Global Params =====================================
    def train(self, step_size, epochs):
        self.step_size = step_size
        self.epochs = epochs
        with Pool(processes=10) as pool:
            results = pool.map(self.single_user_train, range(self.users_nums))
        total_cost = [sum(x) for x in zip(*results)]
        plot_list(total_cost, "epoch", "cost", "HR cost per epoch")

    def single_user_train(self, user):
        cost = []
        optimizer = qml.AdagradOptimizer(stepsize=self.step_size, eps=1e-08)
        kargs = {'user': user}
        for epoch in range(self.epochs):
            self.params[user], kargs = optimizer.step(self.optimize_HR_params, self.params[user], kargs)
            cost.append(kargs['cost'])
        print(f'done user {user}')
        return cost

    def optimize_HR_params(self, params, kargs):
        overall_err = 0
        user=kargs['user']
        target = self.target_probs[user]
        probs = HR_circ_probs(self.qrs_optimized_params, self.user_params[user], params)
        for inter in range(self.items_num):
            overall_err += ((probs[inter] - target[inter])**2)
        kargs['cost'] = overall_err._value.item()

        return overall_err


    def get_recommendation(self, user, items):
        probs = 0
        print(user, self.samples_count)
        # print("reco for user:", user, "samples:", self.samples_count)
        if self.samples_count == "inf":
            probs = HR_circ_probs(self.qrs_optimized_params, self.user_params[user], self.params[user])
        else:
            if self.samples_count == "items_num": shots = self.items_num
            elif self.samples_count == "sqrt"   : shots = int(np.sqrt(self.items_num))
            elif self.samples_count == "log2"   : shots = int(np.log2(self.items_num))
            decimals = []
            for shot in range(shots):
                sample = HR_circ_samples(self.qrs_optimized_params, self.user_params[user], self.params[user])
                decimal = int(''.join(map(str, sample)), 2)
                decimals.append(decimal)

            probs = np.bincount(decimals, minlength=self.items_num)
            probs = probs / np.sum(probs)

        return probs[items]
