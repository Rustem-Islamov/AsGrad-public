import numpy as np
from multiset import Multiset

class Nodes:
    def __init__(self, num_nodes, computing_speeds, oracle, batch_size=None, b=1, delay_type='poisson',
                 server_seed=42):
        self.num_nodes = num_nodes
        self.oracle = oracle
        self.delay_type = delay_type
        max_seed = 424242
        self.bs = batch_size
        self.rng = np.random.default_rng(server_seed)
        self.b = b
        self.computing_speeds = computing_speeds
        self.seeds_workers = [self.rng.choice(max_seed, size=1, replace=False)[0] for _ in range(num_nodes)]
        self.rngs = [np.random.default_rng(seed) for seed in self.seeds_workers]
        self.permutation = np.arange(num_nodes)


    def init(self, x_init):
        np.random.seed(42)
        self.iteration = 0

        self.active_jobs = {}
        self.active_workers = Multiset()
        np.random.shuffle(self.permutation)

        for i in range(self.num_nodes):
            rng = self.rngs[i]
            grad = self.oracle.local_gradient(x_init, i, rng)
            #self.gradients.append(grad)
            self.active_jobs[i] = [(grad, 0)]
            self.active_workers.add(i)

        if self.delay_type == "poisson":
            self.time_left = np.array([np.random.poisson(self.computing_speeds[i])
                                   for i in range(0, self.num_nodes)]).astype(np.float64)

        elif self.delay_type == "normal":
            self.time_left = np.array([np.abs(np.random.normal(self.computing_speeds[i],\
                                                               self.computing_speeds[i])).astype(np.float64) \
                                       + 1 for i in range(0, self.num_nodes)]).astype(np.float64)
        elif self.delay_type == "uniform":
            self.time_left = np.array([np.abs(np.random.uniform(0,\
                                                                self.computing_speeds[i])).astype(np.float64) \
                                       + 1 for i in range(0, self.num_nodes)]).astype(np.float64)
        else:
            self.time_left = np.array(self.computing_speeds).astype(np.float64)

        self.previous_time_left = self.time_left.copy()
        self.last_idx = None

    def is_ready(self):
        return sum(self.time_left <= 0) >= 1


    def decrease_time(self):
        self.previous_time_left = self.time_left.copy()
        while not self.is_ready():
            self.time_left -= 1
        return

    def get_update(self):
        self.decrease_time()
        idx = np.argmin(self.time_left)

        self.time_left[idx] = np.inf
        self.last_idx = idx
        (grad, pi_t) = self.active_jobs[idx][0]
        self.active_jobs[idx].pop(0)
        self.active_workers[idx] -=1

        return grad, self.iteration - pi_t

    def assign_new_job(self, x, assign_type='pure'):
        if assign_type == 'pure':
            rng = self.rngs[self.last_idx]
            self.active_jobs[self.last_idx].append((self.oracle.local_gradient(x, self.last_idx, rng),
                                                   self.iteration))
            self.active_workers[self.last_idx] += 1
            if self.delay_type == "poisson":
                self.time_left[self.last_idx] = np.random.poisson(self.computing_speeds[self.last_idx])
            elif self.delay_type == "normal":
                self.time_left[self.last_idx] = np.abs(np.random.normal(self.computing_speeds[self.last_idx],
                    self.computing_speeds[self.last_idx])).astype(np.int64) + 1
            else:
                self.time_left[self.last_idx] = self.computing_speeds[self.last_idx]

        if assign_type == 'random':
            new_idx = self.rng.choice(range(self.num_nodes))
            rng = self.rngs[new_idx]
            self.active_jobs[new_idx].append((self.oracle.local_gradient(x, self.last_idx, rng),
                                              self.iteration))
            if new_idx not in self.active_workers:
                self.active_workers[new_idx] += 1
                if self.delay_type == "poisson":
                    self.time_left[new_idx] = np.random.poisson(self.computing_speeds[new_idx])
                elif self.delay_type == "normal":
                    self.time_left[new_idx] = np.abs(np.random.normal(self.computing_speeds[new_idx],
                        self.computing_speeds[self.last_idx])).astype(np.int64) + 1
                else:
                    self.time_left[new_idx] = self.computing_speeds[new_idx]

        if assign_type == 'shuffle':
            new_idx = self.permutation[self.iteration%self.num_nodes]
            rng = self.rngs[new_idx]
            self.active_jobs[new_idx].append((self.oracle.local_gradient(x, self.last_idx, rng),
                                              self.iteration))
            if new_idx not in self.active_workers:
                self.active_workers[new_idx] += 1
                if self.delay_type == "poisson":
                    self.time_left[new_idx] = np.random.poisson(self.computing_speeds[new_idx])
                elif self.delay_type == "normal":
                    self.time_left[new_idx] = np.abs(np.random.normal(self.computing_speeds[new_idx],
                        self.computing_speeds[self.last_idx])).astype(np.int64) + 1
                else:
                    self.time_left[new_idx] = self.computing_speeds[new_idx]

        self.iteration += 1
        if self.iteration%self.num_nodes==0:
            self.rng.shuffle(self.permutation)
        self.last_idx = None
        return