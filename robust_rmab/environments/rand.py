import numpy as np
import sys
from ..environments.cohort import Cohort


class Random(Cohort):
    # Constants
    _start_state_probs = np.array([0.5, 0.5])  # TODO: Read this from command line or config file

    def __init__(
        self,
        seed: int,
        num_beneficiaries: int,
        num_states: int,
        min_action_effect: float,
    ):
        """Initialize a cohort of beneficiaries."""
        # Save inputs
        self._num_benefs = num_beneficiaries
        self._num_states = num_states
        self._min_action_effect = min_action_effect

        # Generate transition matrices
        self._mdps = self.get_mdps(self._num_benefs, self._num_states, self._num_actions, self._min_action_effect, seed)

        # Sanity check
        super().__init__()

    def get_mdps(
        self,
        num_benefs: int,
        num_states: int,
        num_actions: int,
        min_action_effect: float,
        seed,
    ) -> np.ndarray:
        # Generate transition matrices
        np.random.seed(seed)
        _mdps = self._generate_mdps(num_benefs, num_states, num_actions, min_action_effect)
        np.random.seed()
        return _mdps

    def _generate_mdps(
        self,
        num_benefs: int,
        num_states: int,
        num_actions: int,
        min_action_effect: float,
    ) -> np.ndarray:
        """Generate transition matrices for a cohort of beneficiaries.

        Args:
            num_benefs (int): Number of beneficiaries in the cohort
            num_states (int): Number of states in each beneficiary's MDP
            num_actions (int): Number of actions in each beneficiary's MDP
            min_action_effect (float): Minimum difference in expected reward between acting and not acting

        Returns:
            np.ndarray: A 4D numpy array of shape (num_benefs, num_states, num_actions, num_states) representing the transition matrices for each beneficiary's MDP.
        """
        # Sanity checks
        assert num_actions == 2
        assert num_states >= 2
        assert num_benefs >= 1
        assert -1 <= min_action_effect < 1

        # Generate random T matrices
        R = np.arange(num_states) / (num_states - 1)
        #   Sample random T matrices
        T = np.random.uniform(size=(num_benefs * num_states, num_actions, num_states))
        T = T / T.sum(axis=-1, keepdims=True)
        #   Resample T matrices that don't have a large enough action effect
        idxs_to_resample = np.where((T[..., 1, :]) @ R < (T[..., 0, :] @ R + min_action_effect))[0]
        #   Resample until all T matrices have a large enough effect
        while len(idxs_to_resample) > 0:
            T[idxs_to_resample] = np.random.uniform(size=T[idxs_to_resample].shape)
            T[idxs_to_resample] = T[idxs_to_resample] / T[idxs_to_resample].sum(axis=-1, keepdims=True)
            idxs_to_resample = np.where(T[..., 1, :] @ R < T[..., 0, :] @ R + min_action_effect)[0]
        #   Reshape T matrices
        T = T.reshape((num_benefs, num_states, num_actions, num_states))

        return T

class RandomHomogeneous(Random):
    def get_mdps(
        self,
        num_benefs: int,
        num_states: int,
        num_actions: int,
        min_action_effect: float,
        seed,
    ) -> np.ndarray:
        # Generate transition matrices
        np.random.seed(seed)
        _mdps = self._generate_mdps(1, num_states, num_actions, min_action_effect)
        _mdps = np.repeat(_mdps, num_benefs, axis=0)
        np.random.seed()
        return _mdps
