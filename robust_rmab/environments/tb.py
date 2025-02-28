import numpy as np
import pandas as pd
from numba import njit, prange
import os
import pickle
from .cohort import Cohort

class TB(Cohort):
    _num_states = 2
    _raw_data_path = "rct_code/domains/data/raw_tb/Patient.csv"
    _processed_data = "rct_code/domains/data/tb.pkl"
    _start_state_probs = np.array([0.5, 0.5])

    def __init__(
        self,
        seed: int,
        num_beneficiaries: int,
        max_effect_size: float,
        prior_coefficient: float,
        min_randomness: float,
    ):
        """Initialize a cohort of beneficiaries."""
        # Get constants from command line
        self._num_benefs = num_beneficiaries
        self._max_effect_size = max_effect_size
        self._prior_coefficient = prior_coefficient
        self._epsilon = min_randomness

        # Get passive transition matrices from data
        if os.path.exists(self._processed_data):
            with open(self._processed_data, "rb") as fr:
                passive_probs = pickle.load(fr)
        else:
            passive_probs = self.generate_T_matrices_from_data(self._raw_data_path)

        # Choose a random _num_benefs beneficiaries and generate action effects
        np.random.seed(seed)
        self._mdps = self.generate_action_effects(passive_probs, self._num_benefs, self._max_effect_size, self._epsilon)
        np.random.seed()

        # Sanity check
        super().__init__()

    def generate_T_matrices_from_data(
        self,
        data_path: str,
    ) -> np.ndarray:
        """Function to generate transition matrices from TB medication adherence data.
        Source: https://github.com/guaguakai/decision-focused-RL/tree/main/TB/

        Args:
            data_path (str): Path to data file.

        Returns:
            np.ndarray: Passive transition matrices. Shape: (num_patients, num_states, num_states)
        """
        # LOAD DATA
        df = pd.read_csv(data_path, parse_dates=['EnrollmentDate'])

        # PROCESS DATA
        df = df[df['EnrollmentDate'] < pd.Timestamp('8/1/17')]  # I don't know why this is necessary

        #  Map `codes' to states
        #   `Codes' in the dataset
        #   1: end date (no distinction for whether a call was made on this date)
        #   4: confirmed taken dose via unshared number
        #   5: confirmed taken dose via shared number
        #   6: missed dose
        #   8: enrollment date (no distinction for whether a call was made on this date)
        #   9: manual dose (didn't receive a call, but some provider marked the patient as having
        #      taken the dose)
        code_dict = {'1': 1, '4': 1, '5': 1, '6': 0, '8': 1, '9': 1}

        #   Convert `codes' to states
        df['AdherenceString'] = df['AdherenceString'].astype(str)
        def convert_to_binary_sequence(sequence):
            return np.array([code_dict[i] for i in sequence])
        df['AdherenceSequence'] = df['AdherenceString'].apply(convert_to_binary_sequence)
        sequences = df['AdherenceSequence'].values

        # Convert adherence values to T matrices
        #   Get counts
        patient_T_matrices = np.array([TB._seq_to_counts(sequence) for sequence in sequences])
        #   Get overall counts
        overall_T_matrix = patient_T_matrices.sum(axis=0)  # Shape: (num_states, num_states)
        #   Get overall probabilities
        overall_T_matrix /= overall_T_matrix.sum(axis=-1, keepdims=True)  # Shape: (num_states, num_states)
        #   Add "prior", i.e., pseudo-counts
        patient_T_matrices += self._prior_coefficient * overall_T_matrix[None, ...]  # Shape: (num_patients, num_states, num_states)
        #   Normalize to get probabilities
        patient_T_matrices /= patient_T_matrices.sum(axis=-1, keepdims=True)  # Shape: (num_patients, num_states, num_states)

        # Save the passive transition matrices
        with open(self._processed_data, "wb") as fw:
            pickle.dump(patient_T_matrices, fw)

        return np.array(patient_T_matrices)

    @staticmethod
    @njit('f8[:,:](i8[:])', fastmath=True)
    def _seq_to_counts(
        sequence: np.ndarray,
    ):
        """Convert a sequence of states to a count matrix.

        Args:
            sequence (np.ndarray): A sequence of states. Shape: (num_timesteps,)

        Returns:
            np.ndarray: A count matrix. Shape: (num_states, num_states)
        """
        # Initialize the count matrix
        count_matrix = np.zeros((2, 2))  # Shape: (num_states, num_states)
        # Count the number of transitions between states
        for i in range(len(sequence) - 1):
            count_matrix[sequence[i], sequence[i + 1]] += 1
        return count_matrix

    @staticmethod
    def generate_action_effects(
        T_passive: np.ndarray,
        num_benefs: int,
        max_effect_size: float,
        epsilon: float,
    ) -> np.ndarray:
        """
        Generate action effects for a set of beneficiaries given a set of passive transition matrices.

        Args:
            T_passive (np.ndarray): Passive transition matrices. Shape: (num_patients, num_states, num_states)
            num_benefs (int): Number of beneficiaries.
            max_effect_size (float): Maximum effect size.
            epsilon (float): Minimum probability of transitioning to any state.

        Returns:
            np.ndarray: Transition matrices. Shape: (num_benefs, num_states, num_actions, num_states)
        """
        # Sample passive transition matrices from the set of passive transition matrices
        patient_idxs = np.random.choice(T_passive.shape[0], size=num_benefs, replace=True)
        T_passive = T_passive[patient_idxs, ...]  # Shape: (num_benefs, num_states, num_states)
        T_active = np.copy(T_passive)

        # Define Action Effects
        #   Patient responds well to call
        benefit_act_00 = np.random.uniform(low=0., high=max_effect_size, size=(num_benefs,))  # will subtract from prob of staying 0,0
        #   Add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition
        benefit_act_11 = benefit_act_00 + np.random.uniform(low=0., high=max_effect_size, size=(num_benefs,))  # will add to prob of staying 1,1
        #   Patient does well on their own, low penalty for not calling
        penalty_pass_11 = np.random.uniform(low=0., high=max_effect_size, size=(num_benefs,))  # will sub from prob of staying 1,1
        penalty_pass_00 = penalty_pass_11 + np.random.uniform(low=0., high=max_effect_size, size=(num_benefs,))  # will add to prob of staying 0,0

        # Apply Action Effects
        #   Add the benefit of acting 
        T_active[..., 0, 0] = np.clip(T_active[..., 0, 0] - benefit_act_00, epsilon, 1 - epsilon)
        T_active[..., 1, 1] = np.clip(T_active[..., 1, 1] + benefit_act_11, epsilon, 1 - epsilon)
        #   Subtract the penalty of not acting
        T_passive[..., 0, 0] = np.clip(T_passive[..., 0, 0] + penalty_pass_00, epsilon, 1 - epsilon)
        T_passive[..., 1, 1] = np.clip(T_passive[..., 1, 1] - penalty_pass_11, epsilon, 1 - epsilon)
        #   Re-Normalise
        T_passive[..., 0, 1] = 1 - T_passive[..., 0, 0]
        T_passive[..., 1, 0] = 1 - T_passive[..., 1, 1]
        T_active[..., 0, 1] = 1 - T_active[..., 0, 0]
        T_active[..., 1, 0] = 1 - T_active[..., 1, 1]

        # Combine the active and passive probabilities
        T_matrices = np.stack([T_passive, T_active], axis=-2)  # Shape: (num_benefs, num_states, num_actions, num_states)

        return T_matrices