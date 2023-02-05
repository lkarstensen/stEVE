from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np


class Device(ABC):
    def __init__(self, diameter: float, total_length: float) -> None:
        self.total_length = total_length
        self.diameter = diameter
        self.rotation = 0.0
        self.inserted_length = 0.0
        self.trans_velocity = 0.0
        self.rot_velocity = 0.0

        self._element_length = None

    @property
    def element_length(self) -> float:
        return self._element_length

    @element_length.setter
    def element_length(self, element_length: float) -> None:
        self._element_length = element_length

    def simu_step(self, dt_simulation: float) -> None:
        new_inserted_length = self.inserted_length + self.trans_velocity * dt_simulation
        if new_inserted_length > self.total_length or new_inserted_length < 0.0:
            self.trans_velocity = 0.0
        else:
            self.inserted_length = new_inserted_length
        self.rotation += self.rot_velocity * dt_simulation

    def reset(self) -> None:
        self.rotation = 0.0
        self.inserted_length = 0.0

    @abstractmethod
    def calc_stiffness_and_damping_changes(
        self, tip_spring_idx: int, last_tip_sping_idx: int
    ) -> Tuple[List, List]:
        ...

    @abstractmethod
    def get_last_inserted_element_stiffness(self) -> float:
        ...

    @abstractmethod
    def get_last_inserted_element_damping(self) -> float:
        ...

    @abstractmethod
    def get_rest_angles_with_stiffnesses(self) -> Tuple[np.ndarray, np.ndarray]:
        ...
