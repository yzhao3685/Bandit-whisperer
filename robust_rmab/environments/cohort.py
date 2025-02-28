import numpy as np
from abc import ABC

class Cohort(ABC):
    # Constants
    _num_actions = 2

    def __init__(self):
        assert hasattr(self, "_num_states"), "Cohort subclasses must define _num_states"
        assert hasattr(self, "_num_actions"), "Cohort subclasses must define _num_actions"
        assert self._num_actions == 2, "Cohort subclasses must have 2 actions"
        assert hasattr(self, "_num_benefs"), "Cohort subclasses must define _num_benefs"
        assert hasattr(self, "_mdps"), "Cohort subclasses must define _mdps"
        assert hasattr(self, "_start_state_probs"), "Cohort subclasses must define _start_state_probs"

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def num_benefs(self) -> int:
        return self._num_benefs

    @property
    def mdps(self) -> np.ndarray:
        return self._mdps

    @property
    def start_state_probs(self) -> np.ndarray:
        return self._start_state_probs
