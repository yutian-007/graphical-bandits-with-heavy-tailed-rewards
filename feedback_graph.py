# -*- coding: utf-8 -*-

from re import L
import numpy as np
from ArmClass import BinomialArm, ParetoArm, StandardTArm


class FeedbackGraph(object):

    def __init__(self, K, np_rng=None):
        if np_rng is None:
            self.np_rng = np.random.RandomState(0)
        else:
            self.np_rng = np_rng

        self.nodes = []
        self.num_nodes = K
        self.edges = None

    def set_np_rng(self, np_rng):
        self.np_rng = np_rng
        for node in self.nodes:
            node.set_np_rng(np_rng)

    def init_nodes(self, distribution='binomial',  parameters=None):
        # create nodes
        self.nodes = []
        K = self.num_nodes
        np_rng = self.np_rng
        if distribution == 'binomial':
            means_list = [0.8, 0.8]
            self.nodes.append(BinomialArm(0, means_list[0], np_rng))
            self.nodes.append(BinomialArm(1, means_list[1], np_rng))

            for id in range(2, K):
                random_mean = means_list[0] - 0.2 + np.random.rand() / 5
                self.nodes.append(BinomialArm(id, random_mean, np_rng))
                means_list.append(random_mean)

        elif distribution == 'pareto':
            a_pareto_list = parameters[0]
            m_pareto_list = parameters[1]
            assert (K == len(a_pareto_list))
            pareto_means_list = []
            for id in range(K):
                self.nodes.append(
                    ParetoArm(id, a_pareto_list[id], m_pareto_list[id], np_rng))
                pareto_means_list.append(self.nodes[id].get_mean)
            print(np.round(pareto_means_list, 3))

        elif distribution == 'standard_t':
            df_standard_t = parameters[0]
            means_list = parameters[1]
            assert (K == len(means_list))
            for id in range(K):
                self.nodes.append(
                    StandardTArm(id, df_standard_t, means_list[id], np_rng))
            print(np.round(means_list, 3))

    def init_edges(self, p=None):
        # create edges
        K = self.num_nodes
        np_rng = self.np_rng
        if p == 0:
            self.edges = np.zeros([K, K])
        elif p > 0 and p <= 1:
            assert (p is not None)
            self.edges = np_rng.uniform(size=[K, K])
            self.edges = (self.edges <= p).astype(int)
            for i in range(self.edges.shape[0]):
                for j in range(i + 1, self.edges.shape[1]):
                    self.edges[j][i] = self.edges[i][j]
        elif p == -1:
            assert (K == 30)
            self.edges = np.zeros([K, K])
            for i in range(4):
                id = 7*i
                for j in range(1, 6):
                    self.edges[id][id+j] = 1
                    self.edges[id+6][id+j] = 1
                    self.edges[id+j][id+(j+1) % 5] = 1
                if i < 2:
                    self.edges[id][K-2] = 1
                    self.edges[id][id+2*7] = 1
                else:
                    self.edges[id][K-1] = 1

            for i in range(self.edges.shape[0]):
                for j in range(i + 1, self.edges.shape[1]):
                    if self.edges[j][i] == 1 or self.edges[i][j] == 1:
                        self.edges[j][i] = 1
                        self.edges[i][j] = 1
        else:
            raise AttributeError

        # add self-loops for all arms
        raw, col = np.diag_indices_from(self.edges)
        self.edges[raw, col] = 1

    def draw(self, index):
        if index >= self.num_nodes:
            raise IndexError
        assert (index >= 0)
        reward = self.nodes[index].sample_reward()
        feedback = [self.nodes[j].sample_reward() if self.edges[index][j] or self.edges[j][index] else None
                    for j in range(self.num_nodes)]
        if self.edges[index][index]:
            feedback[index] = reward
        return reward, feedback

    def best_mean(self):
        result = -np.inf
        idx = 0
        for i, arm in enumerate(self.nodes):
            if arm is not None and arm.get_mean > result:
                result = arm.get_mean
                idx = i
        return idx, result

    def get_neighbors(self, index):
        return [i for i in range(self.num_nodes) if self.edges[index, i] or self.edges[i, index]]

    def get_neighbors_num(self, index):
        return len([i for i in range(self.num_nodes) if self.edges[index, i] or self.edges[i, index]])

    def check_moments_bound(self, r, v):
        # $r$: moment order, $v$: $r$-th moment bound
        nodes = self.nodes
        K = self.num_nodes

        for id in range(K):
            moment = nodes[id].get_moment(r)
            if moment > v:
                print(
                    f'(a,m)={nodes[id].get_parameters_pareto}, epsilon={r-1}, moment={moment}, v={v}')
                assert (0)

    def find_independent_set(self, nodes_list):
        sub_graph = []
        flag_list = [True for _ in range(self.num_nodes)]
        for ii in nodes_list:
            if flag_list[ii]:
                sub_graph.append(ii)
                for jj in self.get_neighbors(ii):
                    flag_list[jj] = False
        return sub_graph
