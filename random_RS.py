import numpy as np

class random_RS():
    def __init__(self,R):
        users_num = len(R)
        items_num = len(R[0])
        self.rand_propbs = []
        for user in range(users_num):
            self.rand_propbs.append(np.random.rand(items_num))

    def get_recommendation(self, user, items):
        return self.rand_propbs[user][items]
