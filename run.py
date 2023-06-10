# -*- coding: utf-8 -*-

from UCB import UCB_N, RUNE_TEM, RUNE_MoM
from AAE import AAE_AlphaSample, RAAE_TEM, RAAE_MoM
import time
from feedback_graph import FeedbackGraph
import matplotlib.pyplot as plt
import numpy as np
import pickle
from multiprocessing import Pool


np.random.seed(42)


def simulate(algorithms):
    cum_regret = np.zeros([len(algorithms), T + 1])
    for ii in range(trials):
        t1 = time.time()
        inst_regret = np.zeros([len(algorithms), T + 1])

        for jj, alg in enumerate(algorithms):
            alg.initialize()
            if alg.__class__.__bases__[0].__name__ == 'AAE_base' or alg.__class__.__bases__[0].__name__ == 'RAAE_base':
                while True:
                    # print(alg.__class__.__bases__[0].__name__)
                    if alg.get_num_active_arm == 1:
                        if alg.select_best_arm() == False:
                            # total round exceed $T$
                            break
                        continue
                    if alg.sample() == False:
                        # total round exceed $T$
                        break
                    alg.eliminate()
                    alg.update_er()
                inst_regret[jj, :] = alg.get_regret

            else:
                # print(alg.__class__.__bases__[0].__name__)
                _, best_mean = alg.graph.best_mean()
                for t in range(1, T + 1):
                    select_id = alg.select_arm()
                    select_mean = alg.get_mean(select_id)
                    inst_regret[jj, t] = best_mean - select_mean
                    alg.observe()

        t2 = time.time()
        print(f'Trials {ii} cost {t2-t1} s', np.sum(inst_regret, axis=1))

        cum_regret += np.cumsum(inst_regret, axis=1)
    return cum_regret / trials


# function for plotting figures
def plot_results(alg_names, cum_regret, distribution, p, epsilon, conf_delta_1, conf_delta_2):
    plt.figure(figsize=(7, 4))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)

    # define linestyle for each algorithm
    all_styles = [':', '--', '-.', '-']
    linestyles = all_styles
    colors = ['g', '#ec2d7a', '#1661ab', 'orange']
    tuples = list(zip(cum_regret, alg_names))
    tuples = sorted(tuples, key=lambda res: res[1])
    cum_regret, alg_names = zip(*tuples)

    for result, name, linestyle, color in zip(cum_regret, alg_names, linestyles, colors):
        plt.plot(result, label=name, linewidth=3.0,
                 linestyle=linestyle, color=color)

    plt.legend(loc='upper left', frameon=True, fontsize=15)
    plt.xlabel('$t$', labelpad=1, fontsize=18)
    plt.ylabel('Cumulative Regret', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, None)
    plt.ylim(-20, None)

    if p is not None and p != -1:
        file_name = f'cum_regret_{K}_{p}_{distribution}_{epsilon}_{conf_delta_1}_{conf_delta_2}.eps'
    else:
        file_name = f'cum_regret_{K}_{distribution}_{epsilon}_{conf_delta_1}__{conf_delta_2}.eps'

    plt.savefig(file_name, dpi=500, bbox_inches='tight')


def run_experiment(distribution='binomial', p=0.0):

    if distribution == "pareto":
        best_mean = 1.0
        other_mean = 0.7
    elif distribution == "standard_t":
        best_mean = 10.0
        other_mean = 7.0
    means_list = []
    best_arm_num = 2
    for _ in range(best_arm_num):
        means_list.append(best_mean)
    for _ in range(best_arm_num, int(K/2)):
        means_list.append(
            other_mean + (best_mean - other_mean)*np.random.random())
    for _ in range(int(K/2), K):
        means_list.append(other_mean * np.random.random())
    np.random.shuffle(means_list)

    parameters = []  # parameters for reward distribution
    moments_bound = 0.0  # (1+\epsilon)-moments bound

    # parameters for pareto distribution
    if distribution == "pareto":
        epsilon = 0.3
        a_pareto_list = [1.1 + epsilon for _ in range(K)]
        m_pareto_list = [means_list[i] *
                         (a_pareto_list[i] - 1) / a_pareto_list[i] for i in range(K)]
        moments_list = [a_pareto_list[i] * m_pareto_list[i] ** (1 + epsilon) /
                        (a_pareto_list[i] - 1 - epsilon) for i in range(K)]
        moments_bound = max(moments_list) + 1e-6
        parameters = [a_pareto_list, m_pareto_list]
    # parameters for student's t-ditribution
    elif distribution == "standard_t":
        df_standard_t = 3.0
        moments_bound = 3.0
        epsilon = 1.0
        parameters = [df_standard_t, means_list]

    # build feedback graph
    graph = FeedbackGraph(K)
    graph.init_nodes(distribution, parameters)
    graph.init_edges(p=p)

    # check $(1+\epsilon)$-th moment is bounded by $v$
    if distribution == "pareto":
        graph.check_moments_bound(1+epsilon, moments_bound)

    # simulate
    algorithms = []
    c1_list = [1.0,0.1,0.01]
    c2_list = [10**(-i) for i in range(3, 10)]
    conf_delta_1 = 0.0001
    conf_delta_2 = 0.0001

    # grid search: UCB-based algorithms
    for c1 in c1_list:
        algorithms.append(UCB_N(T, c1, graph, conf_delta_1))
        if distribution == "pareto":
            algorithms.append(
                RUNE_TEM(T, c1, graph, moments_bound, epsilon, conf_delta_1))
        elif distribution == "standard_t":
            algorithms.append(
                RUNE_MoM(T, c1, graph, moments_bound, epsilon, conf_delta_1))

    # grid search: AAE-based algorithms
    for c2 in c2_list:
        algorithms.append(AAE_AlphaSample(T, c2, graph, conf_delta_2))
        if distribution == "pareto":
            algorithms.append(
                RAAE_TEM(T, c2, graph, moments_bound, epsilon, conf_delta_2))
        elif distribution == "standard_t":
            algorithms.append(
                RAAE_MoM(T, c2, graph, moments_bound, epsilon, conf_delta_2))

    t1 = time.time()
    cum_regrets = simulate(algorithms)
    t2 = time.time()

    alg_names = [alg.__class__.__name__ for alg in algorithms]
    alg_names = [s.replace('_', '-').replace('4', ' ') for s in alg_names]
    alg_list = [
        (alg_names[i], algorithms[i].coeff) for i in range(len(algorithms))]

    results = list(zip(alg_list, np.round(cum_regrets[:, -1], 3)))

    # select the best result in each class of algorithm
    results_couple = sorted(
        enumerate(results), key=lambda res: (res[1][0][0], res[1][1]), reverse=False)
    final_results = []
    final_regrets = []
    final_alg_names = []
    for alg in list(set(alg_names)):
        for res in results_couple:
            if res[1][0][0] == alg:
                final_results.append(res[1])
                final_regrets.append(cum_regrets[res[0], :].reshape(-1))
                final_alg_names.append(alg_list[res[0]][0])
                break
    final_results = sorted(final_results, key=lambda res: res[1])
    print(f'K:{K},p:{p},distribution:{distribution},epsilon:{epsilon},moments_bound:{np.round(moments_bound,3)},conf_delta_1:{conf_delta_1},conf_delta_2:{conf_delta_2},algorithms:{final_results},simulate cost time:{np.round(t2-t1,3)} s')

    plot_results(final_alg_names, final_regrets,
                  distribution, p, epsilon, conf_delta_1, conf_delta_2)
    print('finish plot_results')

    ## save results to .pkl file
    # results_dict = {}
    # results_dict['alg_names'] = final_alg_names
    # results_dict['regrets'] = final_regrets
    # file_name = f'data/cum_regret_{K}_{p}_{distribution}_{epsilon}_{conf_delta_1}__{conf_delta_2}.pkl'
    # with open(file_name, "wb") as tf:
    #     pickle.dump(results_dict, tf)
    # print('save results to file')


def load_from_file(p, distribution, epsilon, conf_delta_1, conf_delta_2):
    # load .pkl file to plot result
    file_name = f'data/cum_regret_{K}_{p}_{distribution}_{epsilon}_{conf_delta_1}__{conf_delta_2}.pkl'
    with open(file_name, "rb+") as tf:
        try:
            results_dict = pickle.load(tf)
        except EOFError:
            print("No this file")
            return None

    final_alg_names = results_dict['alg_names']
    final_regrets = results_dict['regrets']

    for ii in range(len(final_alg_names)):
        alg_name = final_alg_names[ii]
        regret = final_regrets[ii][::2500]
        print(alg_name, np.round(regret, 3))

    plot_results(final_alg_names, final_regrets,
                 distribution, p, epsilon, conf_delta_1, conf_delta_2)


if __name__ == '__main__':
    # configure parameters of experiments
    T = int(1e4)
    trials = 10
    distribution = 'pareto'

    # Random graph generate by Erd˝os-Rényi model
    K = 30
    p = 0.6  # p\in{0.4,0.8}

    # # Deterministic graph constructed by Lu et al. [2021]
    # K = 30
    # p = -1

    run_experiment(distribution, p)
