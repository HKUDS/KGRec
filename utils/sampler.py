import numpy as np
import random

class UniformSampler():
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def sample_negative(self, users, n_items, train_user_dict, sample_num):
        row = sample_num
        col = users.shape[0]
        samples_array = np.zeros((col, row), dtype=np.int64)
        for user_i, user in enumerate(users):
            pos_items = train_user_dict[user]

            for i in range(0, row):
                while True:
                    neg = random.randint(0,n_items-1)
                    if neg not in pos_items:
                        break
                samples_array[user_i, i] = neg
        return samples_array
