# -*- coding: utf-8 -*-
import numpy as np
import abc


class BaseArm(metaclass=abc.ABCMeta):
    def __init__(self, id, mean, np_rng=None):
        # running estimate parameters
        self.muh = 0  # \hat{\mu}
        self.num_selected = 0
        self.num_observed = 0
        self.conf_radius = 0

        # reward information
        self.mean = float(mean)
        self.id = id

        if np_rng is None:
            self.np_rng = np.random.RandomState(0)
        else:
            self.np_rng = np_rng

    def set_np_rng(self, np_rng):
        self.np_rng = np_rng

    @property
    def get_mean(self):
        return self.mean

    @property
    def get_muh(self):
        return self.muh

    @property
    def get_num_selected(self):
        return self.num_selected

    @property
    def get_num_observed(self):
        return self.num_observed

    @property
    def get_conf_radius(self):
        return self.conf_radius

    @property
    def get_id(self):
        return self.id

    @abc.abstractmethod
    def sample_reward(self):
        pass

    def reset(self):
        # reset parameters
        self.muh = 0
        self.num_selected = 0
        self.num_observed = 0
        self.conf_radius = 0

    def update_num(self, new_num_selected, new_num_observed):
        assert (new_num_selected - self.num_selected < 2)
        assert (new_num_observed - self.num_observed < 2)
        self.num_selected = new_num_selected
        self.num_observed = new_num_observed

    def update_muh(self, new_muh):
        self.muh = new_muh

    def set_conf_radius(self, new_conf_radius):
        self.conf_radius = new_conf_radius


class BinomialArm(BaseArm):
    def __init__(self, id, mean=0.5, np_rng=None):
        super().__init__(id, mean, np_rng)

    def sample_reward(self):
        return self.np_rng.binomial(1, self.mean)


class ParetoArm(BaseArm):
    # p(x) = a * m^{a} / x^{a+1} (x>=m)
    def __init__(self, id, a_pareto=2.0, m_pareto=1.0, np_rng=None):
        mean = a_pareto * m_pareto / (a_pareto - 1)
        super().__init__(id, mean, np_rng)
        self.a_pareto = a_pareto
        self.m_pareto = m_pareto

    @property
    def get_parameters_pareto(self):
        return self.a_pareto, self.m_pareto

    def get_moment(self, r):
        assert (self.a_pareto > r)
        return self.a_pareto * self.m_pareto ** r / (self.a_pareto - r)

    def sample_reward(self):
        # np.random.pareto geneate Lomax or Pareto II distribution
        # is a shifted Pareto distribution.
        # The classical Pareto distribution can be obtained from
        # the Lomax distribution by adding $1$ and multiplying by the scale parameter $m$.
        # p(x) = a * m^{a} / x^{a+1} (x>=m)
        return (self.np_rng.pareto(self.a_pareto) + 1) * self.m_pareto


class StandardTArm(BaseArm):
    # p(x) = \frac{\Tau(\frac{df+1}{2})}{\sqrt{\pi*df}\Tau(\frac{df}{2})}*(1+\frac{x^2}{df})^{-\frac{df+1}{2}}
    def __init__(self, id, df_standard_t=2.0, mean=1.0, np_rng=None):
        super().__init__(id, mean, np_rng)
        self.df_standard_t = df_standard_t

    @property
    def get_parameters_standard_t(self):
        return self.df_standard_t

    def sample_reward(self):
        return (self.np_rng.standard_t(self.df_standard_t) + self.mean)
