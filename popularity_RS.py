import random
import numpy as np

class popularity_RS():
    def __init__(self, R):
        counts = np.sum(R, axis=0)
        counts = counts + np.random.uniform(-0.1, 0.1, size=len(R[0])) # make it random
        self.probs = counts/sum(counts)

    def get_recommendation(self, user, items):
        return self.probs[items]

