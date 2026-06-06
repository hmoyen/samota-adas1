import arviz as az
import numpy as np
from scipy.stats import dirichlet

import mdp_simulator.config as config
from mdp_simulator.utils.context import Context, Topics

LOW_REWARD = 5
HIGH_REWARD = 100

EPSILON = 0.001


class Action:
    _probabilities: np.ndarray = None
    _starting_probability: np.ndarray = None
    _outcomes: np.ndarray = None
    _rewards: np.ndarray = None
    _equilibrium_constraints: np.ndarray = None
    # full posterior
    _prior: np.ndarray = None
    _expected: np.ndarray = None
    _hdi: np.ndarray = None
    _constraints_satisfied: bool = None
    # partial posterior
    _prior_queue: np.ndarray = None
    _partial_prior: np.ndarray = None
    _partial_expected: np.ndarray = None
    _partial_posterior_retries: np.ndarray = None
    # utils
    # _priorLock = None
    _constraints_checked: bool = False
    _has_new_values: bool = False
    _is_prior_reset: bool = None
    _is_hdi_exceeded: np.ndarray = None

    # Constructor
    def __init__(self, probability, outcome, reward=HIGH_REWARD) -> None:
        """
        :param probability: float probability (between 1 and 0)
        :param outcome: the outcome state id
        :param reward: the reward of the combination action-outcome
        """
        try:
            assert isinstance(probability, float)
            assert isinstance(outcome, str)
            assert isinstance(reward, int)
        except AssertionError:
            raise AssertionError("Defining an action with inconsistent parameter types.")

        if probability > 1.0 or probability < 0:
            raise ValueError("Adding action with an out of bound probability.")
        self._probabilities = np.array([probability], np.half)
        self._starting_probability = self._probabilities.copy()

        self._outcomes = np.array([outcome], np.str_)
        self._rewards = np.array([reward], float)

    def add_action(self, probability, outcome, reward=HIGH_REWARD) -> None:
        """
        :param probability: float probability (between 1 and 0)
        :param outcome: the outcome state id
        :param reward: the reward of the combination action-outcome
        :return: None
        """
        assert isinstance(probability, float)
        assert isinstance(outcome, str)
        assert isinstance(reward, int)

        if probability > 1.0:
            raise ValueError("Adding action with probability greater than 1")
        elif probability < 0.0:
            raise ValueError("Adding action with negative probability")
        elif 1.0 - EPSILON < self._probabilities.sum() + probability < 1.0 + EPSILON:
            probability = max(1.0 - self._probabilities.sum(), 0)
        elif self._probabilities.sum() + probability >= 1.0 + EPSILON:
            raise ValueError("The total sum of all the actions is greater than 1")

        self._probabilities = np.append(self._probabilities, probability)
        self._probabilities = self._probabilities.astype(np.half)
        self._starting_probability = self._probabilities.copy()

        self._outcomes = np.append(self._outcomes, outcome)
        self._rewards = np.append(self._rewards, reward)

    def add_prior(self, prior: np.ndarray, eq_constraint: np.ndarray) -> None:
        """
        :param prior: A numpy array containing the int params of the Dirichlet distribution
        :param eq_constraint: 2d array with shape (N,2)
        :return: None
        """

        assert isinstance(prior, np.ndarray)
        assert isinstance(eq_constraint, np.ndarray)

        # Check number of arguments: not enough arguments
        if prior.shape[0] != self._probabilities.shape[0] or eq_constraint.shape[0] != self._probabilities.shape[0]:
            raise ValueError("Inconsistencies between prior/equilibrium constraint and probabilities length")

        self.__check_eq_constraints(eq_constraint)
        # Initialize the prior
        # self._priorLock = Lock()
        self._prior = prior
        self._constraints_satisfied = True
        self._equilibrium_constraints = eq_constraint
        self._expected = np.empty(0)
        self._hdi = np.empty(0)
        self._partial_posterior_retries = np.zeros(self._prior.shape, int)
        self._prior_queue = np.ones(self._prior.shape, int)

        self._partial_prior = self._prior
        self._partial_expected = np.empty(0)

        # Compute hdi and expected
        self.compute_posterior()

    def compute_posterior(self):
        # Update all the derived values
        self.__compute_expected_value()
        self.__compute_hdi()
        self.__check_full_and_partial_congruence()
        self._constraints_checked = False
        self._has_new_values = True

    def increment_prior(self, index):
        """
        :param index: An int index of the selected outcome.
        :return: None.
        """
        index = int(index)

        if self._prior is None:
            raise Exception(f"Accessing a not existing Prior")

        # Update the prior values
        if not (self._prior.size > index or index < 0):
            raise Exception(f"Accessing a not existing index of a Prior")

        self._prior[index] += 1

        # create queue element
        queue_element = np.zeros(self._prior.size, int)
        queue_element[index] = 1

        # add queue element to the history and cutoff the excess
        self._prior_queue = np.vstack((queue_element, self._prior_queue))
        self._prior_queue = self._prior_queue[0:config.MAX_HISTORY_PARTIAL_POSTERIOR, :]

        if self._prior_queue.shape[0] < config.MAX_HISTORY_PARTIAL_POSTERIOR:
            self._partial_prior = self._prior
        else:
            self._partial_prior = self._prior_queue.sum(axis=0)

        self.__normalize_prior()

    # Private
    def __check_full_and_partial_congruence(self):
        # Checks that the partial posterior mean is inside the full hdi
        self._is_prior_reset = False
        self._is_hdi_exceeded = np.array([False] * self._partial_expected.size)

        for hdi_interval, partial_expected, index in zip(self._hdi, self._partial_expected,
                                                         range(self._partial_expected.size)):
            disequilibrium = hdi_interval[0] > partial_expected or hdi_interval[1] < partial_expected
            if disequilibrium:
                self._is_hdi_exceeded[index] = True
                self._partial_posterior_retries[index] += 1
            else:
                self._partial_posterior_retries[index] = 0

        if self._partial_posterior_retries.max(initial=None) >= config.MAX_PARTIAL_POSTERIOR_RETRIES:
            self._partial_posterior_retries = np.zeros(self._prior.shape).astype(int)
            self._prior = self._partial_prior
            self._is_prior_reset = True

    def __normalize_prior(self):
        # Makes sure that all the priors have values greater or equal to 1
        for i in range(self._prior.size):
            if self._prior[i] < 1:
                self._prior[i] = 1
            if self._partial_prior[i] < 1:
                self._partial_prior[i] = 1

    def __compute_expected_value(self) -> None:
        # Compute the new value
        self._expected = dirichlet.mean(self._prior)
        # Partial posterior
        self._partial_expected = dirichlet.mean(self._partial_prior)

    def __compute_hdi(self) -> None:
        # Compute the new value
        alpha = self._prior
        sample = np.random.dirichlet(alpha, size=config.HDI_SAMPLES)
        # Transform from (draw,shape) to (chain, draw, shape)
        sample = sample.reshape((-1, sample.shape[0], sample.shape[1]))
        self._hdi = az.hdi(sample, hdi_prob=config.HDI_PROB)

    # Getters
    def get_probabilities(self) -> np.ndarray:
        """
        :return: Numpy array containing all the probabilities.
        """
        return self._probabilities

    def get_starting_probabilities(self) -> np.ndarray:
        """
        :return: Numpy array containing all the starting probabilities (not modified by any semantic space variable)
        """
        return self._starting_probability

    def get_outcomes(self) -> np.ndarray:
        """
        :return: A numpy array with all the state ids of the outcomes
        """
        return self._outcomes

    def get_posterior(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        :return: A tuple with 3 numpy arrays containing respectively Prior, Expected, HDI
        """
        prior = self._prior.copy()
        expected = self._expected.copy()
        hdi = self._hdi.copy()
        is_hdi_exceeded = self._is_hdi_exceeded.copy()

        self._has_new_values = False

        return prior, expected, hdi, is_hdi_exceeded

    def get_rewards(self) -> np.ndarray:
        """
        :return: A numpy array containing the
        """
        return self._rewards

    # Get single elements of the posterior
    def get_expected(self) -> np.ndarray:
        return self._expected

    def get_partial_expected(self) -> np.ndarray:
        return self._partial_expected

    def get_prior(self) -> np.ndarray:
        return self._prior

    def get_partial_prior(self) -> np.ndarray:
        return self._partial_prior

    def get_hdi(self) -> np.ndarray:
        return self._hdi

    def get_eq_constraints(self) -> np.ndarray:
        return self._equilibrium_constraints

    # Setters
    def set_reward(self, outcome, new_reward) -> None:
        index = np.where(self._outcomes == outcome)
        self._rewards[index] = new_reward

    def set_probabilities(self, probabilities: np.ndarray, semantic_space_update=False) -> None:
        """
        :param probabilities: A numpy array containing the new probabilities to set
        :param semantic_space_update: A boolean that specifies whether an update is caused by a semantic space variable.
        :return: None
        """
        assert isinstance(probabilities, np.ndarray)
        assert isinstance(semantic_space_update, bool)

        # Check probability equal 1
        probabilities = probabilities.astype(np.half)
        probabilities = self.__check_probabilities(probabilities)

        # Check same number of probabilities
        if probabilities.shape != self._probabilities.shape:
            raise Exception("Expecting an array with the same shape")

        self._probabilities = probabilities
        # if the update is not caused by a semantic space variable
        #   update the original probability value
        if not semantic_space_update:
            self._starting_probability = self._probabilities.copy()

    # Checks
    def check_constraints_satisfied(self) -> bool:
        """
        :return: A boolean stating whether all the constraints are satisfied or not
        """
        # Check if the posterior changed since last pass (otherwise return last constraint check)
        if self._constraints_checked:
            Context.emit(Topics.DEBUG, "Nothing changed since last constraint check")
            return self._constraints_satisfied
        else:
            self._constraints_checked = True

        self._constraints_satisfied = True
        # The posterior changed since last pass
        for hdi_interval, eqc_interval, outcome, index in zip(self._hdi,
                                                              self._equilibrium_constraints,
                                                              self._outcomes,
                                                              range(self._hdi.size)):
            disequilibrium = hdi_interval[0] > eqc_interval[1] or hdi_interval[1] < eqc_interval[0]
            if disequilibrium:
                self.set_reward(outcome, LOW_REWARD)
                self._constraints_satisfied = False
            else:
                self.set_reward(outcome, HIGH_REWARD)

        return self._constraints_satisfied

    def has_new_values(self) -> bool:
        """
        :return: A boolean stating the presence or absence of new values (since last check) in the posterior
        """
        return self._has_new_values

    def has_prior(self):
        return self._prior is not None

    @staticmethod
    def __check_probabilities(probabilities: np.ndarray) -> np.ndarray:
        """
        :param probabilities: A numpy array containing a set of probabilities.
        :return: A numpy array containing a "normalized" set of probabilities.
        """
        assert isinstance(probabilities, np.ndarray)

        # Check if all values are between 1 and 0
        if probabilities.min(initial=None) < 0 or probabilities.max(initial=None) > 1:
            raise ValueError("Negative values inside a probability")
        # Check if the probability out of bound
        elif probabilities.sum() > 1.0 + EPSILON:
            raise ValueError("The sum of the probabilities has to be equal to 1")
        # Check if probability is close to 1
        elif 1.0 - EPSILON <= probabilities.sum() < 1.0 or 1.0 < probabilities.sum() <= 1.0 + EPSILON:
            # truncate to the fifth decimal
            for i in range(probabilities.size):
                print(probabilities[i])
                probabilities[i] = round(probabilities[i], 5)

            # add or subtract the delta in order to have 1.0 as sum
            delta = 1 - probabilities.sum()
            for i in range(probabilities.size):
                if 0 <= probabilities[i] + delta <= 1.0:
                    probabilities[i] += delta
                    break
        return probabilities

    @staticmethod
    def __check_eq_constraints(eq_constraints: np.ndarray) -> None:
        """
        :param eq_constraints: A numpy array containing a set of equilibrium constraints.
        """
        assert isinstance(eq_constraints, np.ndarray)
        assert len(eq_constraints.shape) == 2
        assert eq_constraints.shape[1] == 2

        # Check values between 0 and 1
        if eq_constraints.max(initial=None) > 1.0 or eq_constraints.min(initial=None) < 0.0:
            raise ValueError("Equilibrium constraints have to be between 0 and 1.")

        # Check row by row if the first element is greater than the second
        for i in range(eq_constraints.shape[0]):
            if eq_constraints[i][0] > eq_constraints[i][1]:
                raise ValueError("Equilibrium constraints must have a lower and upper bound.")
