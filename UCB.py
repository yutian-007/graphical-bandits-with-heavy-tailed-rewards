# -*- coding: utf-8 -*-

import numpy as np
import abc
import copy

np.random.seed(42)


class UCB_base(metaclass=abc.ABCMeta):  # abstract class of `UCB`
    def __init__(self, T, coeff, graph, delta):
        self.T = T
        self.t = 1
        self.coeff = coeff
        self.graph = copy.deepcopy(graph)
        self.delta = delta

        self.num_arms = graph.num_nodes
        self.pulled_idx = None
        self.active_arms = None

        self.DEBUG = False

    def initialize(self):
        self.t = 1
        self.active_arms = []
        for arm in self.graph.nodes:
            arm.reset()

    def set_np_rng(self, np_rng):
        self.graph.set_np_rng(np_rng)

    def select_arm(self):
        self.t += 1
        graph = self.graph

        for i in range(self.num_arms):
            if i not in self.active_arms:
                assert (self.graph.nodes[i].get_num_observed == 0)
                self.active_arms.append(i)
                self.pulled_idx = i
                return self.pulled_idx

        # update conf_radius for all active arms
        for id in self.active_arms:
            arm = graph.nodes[id]
            assert (arm.get_num_observed > 0)
            new_conf_radius = self.compute_conf_radius(arm)
            arm.set_conf_radius(new_conf_radius)

        # compute UCB index for all active arms
        coeff = self.coeff
        score = [(graph.nodes[id].get_muh + coeff * graph.nodes[id].get_conf_radius)
                 for id in self.active_arms]

        # select the arm with maximum UCB index
        self.pulled_idx = self.active_arms[np.argmax(score)]

        if self.DEBUG == True:
            muhs = [(graph.nodes[id].get_muh) for id in self.active_arms]
            conf_radius = [(graph.nodes[id].get_conf_radius)
                           for id in self.active_arms]
            means = [(graph.nodes[id].get_mean)
                     for id in self.active_arms]
            if self.t < 100:
                print(self.pulled_idx)
                print("arms:", self.active_arms)
                print("means:", np.round(means, 3))
                print("muh:", np.round(muhs, 3))
                print("conf_radius:", np.round(conf_radius, 3))

        return self.pulled_idx

    def update_selected_arm(self, reward):
        idx = self.pulled_idx
        graph = self.graph

        num_selected = graph.nodes[idx].get_num_selected
        num_observed = graph.nodes[idx].get_num_observed

        new_num_selected = num_selected + 1
        new_num_observed = num_observed + 1
        assert (new_num_observed >= new_num_selected)
        graph.nodes[idx].update_num(new_num_selected, new_num_observed)

        new_muh = self.update_mean_estimator(reward, graph.nodes[idx])
        graph.nodes[idx].update_muh(new_muh)

    def update_neighbor_arm(self, feedback):
        idx = self.pulled_idx
        graph = self.graph

        for id in range(len(feedback)):
            # skip the chosen arm
            if id == idx:
                continue

            if feedback[id] is not None:
                if id not in self.active_arms:
                    self.active_arms.append(id)

                observed_reward = feedback[id]

                num_selected = graph.nodes[id].get_num_selected
                num_observed = graph.nodes[id].get_num_observed

                new_num_selected = num_selected
                new_num_observed = num_observed + 1
                assert (new_num_observed >= new_num_selected)
                graph.nodes[id].update_num(
                    new_num_selected, new_num_observed)

                new_muh = self.update_mean_estimator(
                    observed_reward, graph.nodes[id])
                graph.nodes[id].update_muh(new_muh)

    @abc.abstractmethod
    def compute_conf_radius(self, arm):
        pass

    @abc.abstractmethod
    def observe(self, t):
        pass

    @abc.abstractmethod
    def update_mean_estimator(self, reward, arm):
        pass

    def get_mean(self, id):
        assert (id < self.num_arms)
        return self.graph.nodes[id].get_mean


class UCB_N(UCB_base):
    def __init__(self, T, coeff, graph, delta):
        super().__init__(T, coeff, graph, delta)

    def compute_conf_radius(self, arm):
        # update conf_radius for all arms
        assert (arm.get_num_observed > 0)
        conf_radius = np.sqrt(np.log(1 / self.delta) /
                              (2 * arm.get_num_observed))
        return conf_radius

    def update_mean_estimator(self, reward, arm):
        old_muh = arm.get_muh
        num_observed = arm.get_num_observed
        assert (num_observed > 0)
        new_muh = (old_muh * (num_observed - 1) + reward) / \
            (num_observed)
        return new_muh

    def observe(self):
        reward, feedback = self.graph.draw(self.pulled_idx)
        if self.DEBUG == True:
            if self.t < 100:
                print(self.pulled_idx, reward)
        self.update_selected_arm(reward)
        self.update_neighbor_arm(feedback)


# Algorithms for Heavy-Tailed Bandits
class RUNE_TEM(UCB_base):
    def __init__(self, T, coeff, graph, v, epsilon, delta):
        super().__init__(T, coeff, graph, delta)
        self.v = float(v)
        self.epsilon = float(epsilon)

    def compute_conf_radius(self, arm):
        # update conf_radius for all arms
        assert (arm.get_num_observed > 0)
        N_i = self.graph.get_neighbors_num(arm.get_id)

        conf_radius = 5 * \
            self.v ** (1 / (1 + self.epsilon)) * (np.log(N_i / self.delta) /
                                                  (arm.get_num_observed)) ** (self.epsilon / (1 + self.epsilon))
        return conf_radius

    def update_mean_estimator(self, reward, arm):
        old_muh = arm.get_muh
        num_observed = arm.get_num_observed
        id = arm.get_id
        N_i = self.graph.get_neighbors_num(id)
        B_t = (self.v * num_observed / (np.log(N_i / self.delta))
               ) ** (1 / (1 + self.epsilon))
        truncated_reward = reward * (abs(reward) <= B_t)
        assert (num_observed > 0)
        new_muh = (old_muh * (num_observed - 1) + truncated_reward) / \
            (num_observed)
        return new_muh

    def observe(self):
        reward, feedback = self.graph.draw(self.pulled_idx)
        self.update_selected_arm(reward)
        self.update_neighbor_arm(feedback)


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


class RUNE_MoM(UCB_base):
    def __init__(self, T, coeff, graph, v, epsilon, delta):
        super().__init__(T, coeff, graph, delta)
        self.v = float(v)
        self.epsilon = float(epsilon)

    def initialize(self):
        self.t = 1
        self.active_arms = []
        self.history = []
        for arm in self.graph.nodes:
            arm.reset()
            self.history.append([])

    def compute_conf_radius(self, arm):
        # update conf_radius for all arms
        assert (arm.get_num_observed > 0)
        N_i = self.graph.get_neighbors_num(arm.get_id)

        return (12 * self.v) ** (1 / (1 + self.epsilon)) * \
            (8 * np.log(N_i / self.delta) / (arm.get_num_observed)
             ) ** (self.epsilon / (1+self.epsilon))

    def update_mean_estimator(self, reward, arm):
        id = arm.get_id
        N_i = self.graph.get_neighbors_num(id)
        self.history[id].append(reward)
        max_block_num = len(self.history[id]) / 2
        block_num = int(np.ceil(min(max_block_num,
                                    8 * np.log(N_i / self.delta) - 1)))
        new_muh = MoM(self.history[id], block_num)
        return new_muh

    def observe(self):
        reward, feedback = self.graph.draw(self.pulled_idx)
        self.update_selected_arm(reward)
        self.update_neighbor_arm(feedback)
