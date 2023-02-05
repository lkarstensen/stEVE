import math
from typing import List, Tuple
from .device import Device, np


class JWire(Device):
    def __init__(
        self,
        tip_length: float = 15.2,
        total_length: float = 500,
        tip_angle: float = math.pi * 0.4,
        flex_length: float = 15.2,
        flex_rotary_stiffness: float = 7e4,
        flex_rotary_damping: float = 1,
        stiff_rotary_stiffness: float = 1e5,
        stiff_rotary_damping: float = 1,
        diameter: int = 0.9,
    ) -> None:
        super().__init__(diameter=diameter, total_length=total_length)
        self.tip_length = tip_length
        self.tip_angle = tip_angle
        self.flex_length = flex_length
        self.flex_rotary_stiffness = flex_rotary_stiffness
        self.flex_rotary_damping = flex_rotary_damping
        self.stiff_rotary_stiffness = stiff_rotary_stiffness
        self.stiff_rotary_damping = stiff_rotary_damping

        self._n_springs_tip = 0
        self._n_springs_total = 0
        self._rotary_spring_stiffnesses_all = []
        self._rotary_spring_dampings_all = []

    @property
    def element_length(self) -> float:
        return self._element_length

    @element_length.setter
    def element_length(self, element_length: float) -> None:
        self._element_length = element_length
        self._n_springs_total = int(self.total_length / element_length)
        self._n_springs_tip = int(self.tip_length / element_length)
        self._n_springs_flex = int(self.flex_length / element_length)
        n_springs_stiff = self._n_springs_total - self._n_springs_flex
        self._rotary_spring_stiffnesses_all = [
            self.flex_rotary_stiffness
        ] * self._n_springs_flex + [self.stiff_rotary_stiffness] * n_springs_stiff
        self._rotary_spring_dampings_all = [
            self.flex_rotary_damping
        ] * self._n_springs_flex + [self.stiff_rotary_damping] * n_springs_stiff

    def simu_step(self, dt_simulation: float) -> None:
        super().simu_step(dt_simulation)

    def calc_stiffness_and_damping_changes(
        self, tip_spring_idx: int, last_tip_spring_idx: int
    ) -> Tuple[List, List]:

        stiffness_change = []
        damping_change = []

        if tip_spring_idx == last_tip_spring_idx:
            ...
        elif tip_spring_idx < last_tip_spring_idx:
            stiffness_change.append([tip_spring_idx, self.flex_rotary_stiffness])
            damping_change.append([tip_spring_idx, self.flex_rotary_damping])
            if self.inserted_length > self.flex_length + self.element_length:
                if self.flex_rotary_stiffness != self.stiff_rotary_stiffness:
                    stiffness_change.append(
                        [
                            tip_spring_idx + self._n_springs_flex,
                            self.stiff_rotary_stiffness - self.flex_rotary_stiffness,
                        ]
                    )
                if self.flex_rotary_damping != self.stiff_rotary_damping:
                    damping_change.append(
                        [
                            tip_spring_idx + self._n_springs_flex,
                            self.stiff_rotary_damping - self.flex_rotary_damping,
                        ]
                    )
        else:
            stiffness_change.append([last_tip_spring_idx, -self.flex_rotary_stiffness])
            damping_change.append([last_tip_spring_idx, -self.flex_rotary_damping])
            if self.inserted_length > self.flex_length + self.element_length:
                if self.flex_rotary_stiffness != self.stiff_rotary_stiffness:
                    stiffness_change.append(
                        [
                            last_tip_spring_idx + self._n_springs_flex,
                            self.flex_rotary_stiffness - self.stiff_rotary_stiffness,
                        ]
                    )
                if self.flex_rotary_damping != self.stiff_rotary_damping:
                    damping_change.append(
                        [
                            last_tip_spring_idx + self._n_springs_flex,
                            self.flex_rotary_damping - self.stiff_rotary_damping,
                        ]
                    )

        return stiffness_change, damping_change

    def get_last_inserted_element_stiffness(self) -> float:
        if self.inserted_length <= self.flex_length:
            return self.flex_rotary_stiffness
        else:
            return self.stiff_rotary_stiffness

    def get_last_inserted_element_damping(self) -> float:
        if self.inserted_length <= self.flex_length:
            return self.flex_rotary_damping
        else:
            return self.stiff_rotary_damping

    def get_rest_angles_with_stiffnesses(self) -> Tuple[np.ndarray, np.ndarray]:

        spring_rest_angle = (
            self.tip_angle * math.sin(self.rotation) / self._n_springs_tip
        )
        n_springs = int(self.inserted_length / self.element_length)
        n_springs = min(n_springs, self._n_springs_tip)
        rest_angles = np.ones((n_springs)) * spring_rest_angle

        if n_springs <= self._n_springs_flex:
            stiffnesses = np.ones((n_springs)) * self.flex_rotary_stiffness
        else:
            flex = np.ones((self._n_springs_flex)) * self.flex_rotary_stiffness
            stiff = (
                np.ones((n_springs - self._n_springs_flex))
                * self.stiff_rotary_stiffness
            )
            stiffnesses = np.concatenate((flex, stiff), axis=0)

        return rest_angles, stiffnesses
