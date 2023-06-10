# -*- coding: utf-8 -*-

import numpy as np
import abc
import copy

np.random.seed(42)


class AAE_base(metaclass=abc.ABCMeta):  # abstract class of `AAE`
    def __init__(self, T, coeff, graph, delta, np_rng=None):
        if np_rng is None:
            self.np_rng = np.random.RandomState(0)
        else:
            self.np_rng = np_rng
        self.T = T
        self.coeff = coeff
        self.graph = copy.deepcopy(graph)
        self.K = graph.num_nodes
        self.id_star = None
        self.active_arms = None
        self.er = None
        self.nr = None
        self.t = None
        self.best_id, self.best_mean = graph.best_mean()
        self.delta = delta  # default； 1 / (2*self.K*self.T)

    def initialize(self):
        self.regret_list = np.zeros(self.T + 1)
        self.t = 1
        self.er = 1 / 4 ** self.epsilon
        self.active_arms = list(range(self.K))
        for arm in self.graph.nodes:
            arm.reset()

    def set_np_rng(self, np_rng):
        self.np_rng = np_rng
        self.graph.set_np_rng(np_rng)

    def eliminate(self):
        means_list = [self.graph.nodes[id].get_muh for id in self.active_arms]
        mu_star = np.max(means_list)

        # active arm elimination
        self.active_arms = [id for id in self.active_arms if self.graph.nodes[id].get_muh >=
                            mu_star - 2*self.er]

        assert (len(self.active_arms) > 0)

    def update_er(self):
        self.er /= 2**self.epsilon

    @abc.abstractmethod
    def compute_nr(self):
        pass

    @abc.abstractmethod
    def update_mean_estimator(self, reward, arm):
        pass

    @abc.abstractmethod
    def sample(self):
        pass

    @property
    def get_regret(self):
        return self.regret_list

    @property
    def get_num_active_arm(self):
        return len(self.active_arms)

    def select_best_arm(self):
        assert (len(self.active_arms) == 1)
        if self.t > self.T:
            return False

        graph = self.graph
        arm_id = self.active_arms[0]
        select_mean = graph.nodes[arm_id].get_mean
        self.regret_list[self.t] = self.best_mean - select_mean
        self.t += 1
        return True


class AAE_AlphaSample(AAE_base):
    def __init__(self, T, coeff, graph, delta):
        super().__init__(T, coeff, graph, delta)
        self.epsilon = 1.0

    def compute_nr(self):
        res = int(self.coeff * np.ceil(2 *
                  np.log(1 / self.delta) / self.er ** 2))
        return res

    def update_mean_estimator(self, reward, arm):
        old_muh = arm.get_muh
        num_observed = arm.get_num_observed
        assert (num_observed > 0)
        new_muh = (old_muh * (num_observed - 1) + reward) / \
            (num_observed)
        return new_muh

    def alphaSample(self):
        assert (len(self.active_arms) > 1)
        if self.t > self.T:
            return False

        graph = self.graph
        U = copy.deepcopy(self.active_arms)

        while len(U) > 0:
            if self.t > self.T:
                return False
            uid = self.np_rng.randint(0, len(U))
            arm_id = U[uid]
            select_mean = graph.nodes[arm_id].get_mean
            _, feedback = graph.draw(arm_id)
            self.regret_list[self.t] = self.best_mean - select_mean
            self.t += 1

            for id in range(len(feedback)):
                if feedback[id] is not None and id in self.active_arms:
                    if id in U:
                        U.remove(id)

                    obeserved_reward = feedback[id]

                    num_selected = graph.nodes[id].get_num_selected
                    num_observed = graph.nodes[id].get_num_observed

                    if id == arm_id:
                        new_num_selected = num_selected + 1
                    else:
                        new_num_selected = num_selected
                    new_num_observed = num_observed + 1
                    assert (new_num_observed >= new_num_selected)
                    graph.nodes[id].update_num(
                        new_num_selected, new_num_observed)

                    new_muh = self.update_mean_estimator(
                        obeserved_reward, graph.nodes[id])
                    graph.nodes[id].update_muh(new_muh)

        return True

    def sample(self):
        assert (len(self.active_arms) > 1 and self.t <= self.T)
        self.nr = self.compute_nr()
        for _ in range(self.nr):
            if self.alphaSample() == False:
                # total round exceed $T$
                return False
        return True


class RAAE_base(AAE_base):
    def __init__(self, T, coeff, graph, v, epsilon, delta):
        super().__init__(T, coeff, graph, delta)
        self.v = float(v)
        self.epsilon = float(epsilon)

    @abc.abstractmethod
    def compute_nr(self):
        pass

    @abc.abstractmethod
    def update_mean_estimator(self, reward, arm):
        pass

    def sample_once(self, independent_set):
        assert (len(self.active_arms) > 1)
        if self.t > self.T:
            return False

        for arm_id in independent_set:
            if self.t > self.T:
                return False

            assert (arm_id in self.active_arms)
            select_mean = self.graph.nodes[arm_id].get_mean
            _, feedback = self.graph.draw(arm_id)
            self.regret_list[self.t] = self.best_mean - select_mean
            self.t += 1

            for id in range(len(feedback)):
                if feedback[id] is not None and id in self.active_arms:

                    observed_reward = feedback[id]

                    num_selected = self.graph.nodes[id].get_num_selected
                    num_observed = self.graph.nodes[id].get_num_observed

                    if id == arm_id:
                        new_num_selected = num_selected + 1
                    else:
                        new_num_selected = num_selected
                    new_num_observed = num_observed + 1
                    assert (new_num_observed >= new_num_selected)
                    self.graph.nodes[id].update_num(
                        new_num_selected, new_num_observed)

                    new_muh = self.update_mean_estimator(
                        observed_reward, self.graph.nodes[id])
                    self.graph.nodes[id].update_muh(new_muh)

        return True

    def sample(self):
        assert (len(self.active_arms) > 1 and self.t <= self.T)
        if len(self.active_arms) > 2:
            independent_set = self.graph.find_independent_set(self.active_arms)
        else:
            independent_set = self.active_arms

        self.nr = self.compute_nr()
        for _ in range(self.nr):
            if self.sample_once(independent_set) == False:
                return False
        return True


class RAAE_TEM(RAAE_base):
    def __init__(self, T, coeff, graph, v, epsilon, delta):
        super().__init__(T, coeff, graph, v, epsilon, delta)

    def compute_nr(self):
        res = int(self.coeff * np.ceil(5 * (5*self.v)**(1/self.epsilon) * np.log(1 / self.delta)) /
                  (self.er)**((1+self.epsilon)/self.epsilon))
        return res

    def update_mean_estimator(self, reward, arm):
        old_muh = arm.get_muh
        num_observed = arm.get_num_observed
        B_t = (self.v * num_observed / (np.log(1 / self.delta))
               ) ** (1 / (1 + self.epsilon))
        truncated_reward = reward * (abs(reward) <= B_t)
        assert (num_observed > 0)
        new_muh = (old_muh * (num_observed - 1) + truncated_reward) / \
            (num_observed)
        return new_muh


def MoM(rewards_history, block_num):
    total_size = len(rewards_history)
    block_size = int(np.ceil(total_size / block_num))
    means = np.zeros(block_num)
    for ii in range(block_num):
        startId = ii * block_size
        endId = min(total_size, (ii + 1) * block_size)
        if startId < endId:
            means[ii] = np.mean(rewards_history[startId:endId])
    return np.median(means)


class RAAE_MoM(RAAE_base):
    def __init__(self, T, coeff, graph, v, epsilon, delta):
        super().__init__(T, coeff, graph, v, epsilon, delta)
        self.history = []

    def initialize(self):
        self.regret_list = np.zeros(self.T + 1)
        self.t = 1
        self.er = 1 / 4 ** self.epsilon
        self.active_arms = list(range(self.K))
        self.history = []
        for arm in self.graph.nodes:
            arm.reset()
            self.history.append([])

    def compute_nr(self):
        res = int(self.coeff * np.ceil((12*self.v)**(1 / self.epsilon) *
                                       (8 * np.log(1 / self.delta) + 1)) /
                  (self.er)**((1 + self.epsilon) / self.epsilon))
        return res

    def update_mean_estimator(self, reward, arm):
        id = arm.get_id
        self.history[id].append(reward)
        max_block_num = len(self.history[id]) / 2
        block_num = int(np.ceil(min(max_block_num,
                                    8 * np.log(1 / self.delta))))
        new_muh = MoM(self.history[id], block_num)
        return new_muh
