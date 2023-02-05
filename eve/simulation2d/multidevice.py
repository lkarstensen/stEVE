from copy import deepcopy
import math
import pymunk
from typing import List, Tuple
import numpy as np

from .simulation2d import Simulation2D
from ..vesseltree import VesselTree
from ..pathfinder import Pathfinder
from .device import Device


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


class MultiDevice(Simulation2D):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device],
        velocity_limits: List[Tuple[float, float]],
        element_length: int = 3,
        dimension_to_omit: str = "y",
        stop_device_at_tree_end: bool = True,
        image_frequency: float = 7.5,
        dt_simulation: float = 2.5 / 10000,
        friction: float = 1.0,
        damping: float = 0.000001,
        body_mass: float = 0.01,
        body_moment: float = 0.1,
        linear_stiffness: float = 2.5e6,
        linear_damping: float = 100,
        last_segment_kp_angle: float = 3,
        last_segment_kp_translation: float = 5,
        tracking_low_custom: Tuple[float, float, float] = None,
        tracking_high_custom: Tuple[float, float, float] = None,
    ) -> None:

        super().__init__(
            vessel_tree=vessel_tree,
            devices=devices,
            velocity_limits=velocity_limits,
            dimension_to_omit=dimension_to_omit,
            image_frequency=image_frequency,
            dt_simulation=dt_simulation,
            damping=damping,
            last_segment_kp_angle=last_segment_kp_angle,
            last_segment_kp_translation=last_segment_kp_translation,
            tracking_low_custom=tracking_low_custom,
            tracking_high_custom=tracking_high_custom,
        )
        self.element_length = element_length
        self.friction = friction
        self.stop_device_at_tree_end = stop_device_at_tree_end

        self.body_mass = body_mass
        self.body_moment = body_moment
        self.linear_stiffness = linear_stiffness
        self.linear_damping = linear_damping
        self.last_segment_kp_angle = last_segment_kp_angle
        self.last_segment_kp_translation = last_segment_kp_translation

        self._segment_bodies: List[pymunk.Body] = []
        self._segment_shapes: List[pymunk.shapes.Segment] = []
        self._joints: List[pymunk.Constraint] = []
        self._springs: List[pymunk.DampedRotarySpring] = []
        self._stiffnesses: np.ndarray
        self._combined_inserted_length: float
        self._devices_first_spring_idx: List[int]
        self._last_devices_first_spring_idx: List[int]
        self._n_springs_last_step: int
        self._centerline_tree = None
        for device in self.devices:
            device.element_length = element_length

    @property
    def device_rotations(self) -> List[float]:
        rotations = [device.rotation for device in self.devices]
        return rotations

    @property
    def tracking_per_device(self) -> List[np.ndarray]:
        tracking = self.tracking
        tracking_per_device = [tracking[idx:] for idx in self._devices_first_spring_idx]
        return tracking_per_device

    @property
    def device_lengths_inserted(self) -> List[float]:
        return [device.inserted_length for device in self.devices]

    @property
    def device_lengths_maximum(self) -> List[float]:
        return [device.total_length for device in self.devices]

    @property
    def device_diameters(self) -> List[float]:
        return [device.diameter for device in self.devices]

    def step(self, action: np.ndarray) -> None:
        action = action.reshape(self.action_high.shape)
        action = np.clip(action, -self.velocity_limits, self.velocity_limits)
        inserted_lengths = self.device_lengths_inserted

        mask = np.where(inserted_lengths + action[:, 0] / self.image_frequency <= 0.0)
        action[mask, 0] = 0.0

        tip = self.tracking[0]
        if self.stop_device_at_tree_end and self.vessel_tree.at_tree_end(tip):
            max_length = max(inserted_lengths)
            if max_length > 10:
                dist_to_longest = max_length - inserted_lengths
                movement = action[:, 0] / self.image_frequency
                mask = np.argwhere(movement > dist_to_longest)
                action[mask, 0] = 0.0
        self._last_action = action

        for i in range(action.shape[0]):
            self.devices[i].trans_velocity = action[i, 0]
            self.devices[i].rot_velocity = action[i, 1]

        for _ in range(int((1 / self.image_frequency) / self.dt_simulation)):
            longest_device = self.devices[0]
            for device in self.devices:
                device.simu_step(self.dt_simulation)
                if device.inserted_length > longest_device.inserted_length:
                    longest_device = device

            self._combined_inserted_length = longest_device.inserted_length
            for i in range(len(self.devices)):
                dist_to_combined_tip = (
                    self._combined_inserted_length - self.devices[i].inserted_length
                )
                self._devices_first_spring_idx[i] = int(
                    dist_to_combined_tip / self.element_length
                )
            self._add_or_remove_element()
            self._set_base_translation(longest_device.trans_velocity)
            self._set_insertion_point_stabilisation()
            self._set_stiffness_and_damping()
            self._set_rotation()
            self._last_devices_first_spring_idx = deepcopy(
                self._devices_first_spring_idx
            )
            self._n_springs_last_step = len(self._springs)

            self._space.step(self.dt_simulation)

    def _remove_segments_and_constraints(self):
        for joint_spring in zip(self._joints, self._springs):
            self._space.remove(*joint_spring)
        self._space.remove(*self._segment_shapes)
        self._space.remove(*self._segment_bodies)
        del self._joints
        del self._springs
        del self._segment_shapes
        del self._segment_bodies
        self._segment_bodies = []
        self._segment_shapes = []
        self._springs = []
        self._joints = []

    def reset_devices(self):
        self._remove_segments_and_constraints()
        for device in self.devices:
            device.reset()
        self._devices_first_spring_idx: List[int] = [0] * len(self.devices)
        self._last_devices_first_spring_idx: List[int] = [0] * len(self.devices)
        self._combined_inserted_length = 0.0
        self._n_springs_last_step = 0
        self._stiffnesses = np.array([])
        self._last_rest_angles = np.array([])
        self._add_first_element()

    def _add_or_remove_element(self):
        n_elements = len(self._segment_bodies)
        d_n_elements = int(self._combined_inserted_length / self.element_length) + 1
        if d_n_elements > n_elements:
            self._add_element()

        elif d_n_elements < n_elements:
            self._remove_element()

    def _add_first_element(self):
        ip = pymunk.Vec2d(*self.insertion_point)
        ip_dir = pymunk.Vec2d(*self.insertion_direction)

        wire_segment_body = pymunk.Body(mass=self.body_mass, moment=self.body_moment)
        wire_segment_body.position = ip - ip_dir * self.element_length

        wire_segment_body.angle = 0.0
        wire_segment_shape = pymunk.shapes.Segment(
            wire_segment_body,
            pymunk.Vec2d(0, 0),
            ip_dir * self.element_length,
            0.1,
        )
        wire_segment_shape.filter = pymunk.ShapeFilter(group=1)
        self._space.add(wire_segment_body, wire_segment_shape)
        self._segment_bodies.append(wire_segment_body)
        self._segment_shapes.append(wire_segment_shape)

    def _add_element(self):
        ip_dir = pymunk.Vec2d(*self.insertion_direction)
        segment_position = self._segment_bodies[-1].position
        segment_angle = self._segment_bodies[-1].angle
        wire_segment_body = pymunk.Body(mass=self.body_mass, moment=self.body_moment)
        wire_segment_body.position = segment_position - ip_dir * self.element_length
        wire_segment_body.angle = segment_angle
        wire_segment_shape = pymunk.shapes.Segment(
            wire_segment_body,
            pymunk.Vec2d(0, 0),
            ip_dir * self.element_length,
            self._segment_shapes[-1].radius,
        )
        wire_segment_shape.filter = pymunk.ShapeFilter(group=1)

        joint = pymunk.DampedSpring(
            self._segment_bodies[-1],
            wire_segment_body,
            (0, 0),
            self._segment_bodies[-1].position - wire_segment_body.position,
            0.0,
            self.linear_stiffness,
            self.linear_damping,
        )

        stiffness = 0.0
        damping = 0.0
        for device in self.devices:
            stiffness += device.get_last_inserted_element_stiffness()
            damping += device.get_last_inserted_element_damping()

        spring = pymunk.DampedRotarySpring(
            self._segment_bodies[-1], wire_segment_body, 0.0, stiffness, damping
        )
        spring.collide_bodies = False
        self._space.add(wire_segment_body, wire_segment_shape, joint, spring)
        self._segment_bodies.append(wire_segment_body)
        self._segment_shapes.append(wire_segment_shape)
        self._joints.append(joint)
        self._springs.append(spring)
        self._stiffnesses = np.append(self._stiffnesses, stiffness)
        self._last_rest_angles = np.append(self._last_rest_angles, 0.0)

    def _remove_element(self):
        if self._joints:
            joint = self._joints.pop(-1)
            spring = self._springs.pop(-1)
            self._space.remove(joint, spring)
            body = self._segment_bodies.pop(-1)
            shape = self._segment_shapes.pop(-1)
            self._space.remove(body, shape)
            self._stiffnesses = self._stiffnesses[:-1]
            self._last_rest_angles = self._last_rest_angles[:-1]

    def _set_stiffness_and_damping(self):
        for i in range(len(self.devices)):
            idx = self._devices_first_spring_idx[i]
            last_idx = self._last_devices_first_spring_idx[i]
            if (
                last_idx >= self._n_springs_last_step and idx >= len(self._springs)
            ) or last_idx == idx:
                stiffness_changes = []
                damping_changes = []
            else:
                device = self.devices[i]
                (
                    stiffness_changes,
                    damping_changes,
                ) = device.calc_stiffness_and_damping_changes(idx, last_idx)

            for stiffness_change in stiffness_changes:
                spring_idx = stiffness_change[0]
                change = stiffness_change[1]
                self._springs[spring_idx].stiffness += change

            for damping_change in damping_changes:
                spring_idx = damping_change[0]
                change = damping_change[1]
                self._springs[spring_idx].damping += change

    def _set_rotation(self):
        combined_rest_angles = np.zeros_like(self._stiffnesses)
        for i in range(len(self.devices)):
            tip_idx = self._devices_first_spring_idx[i]
            rest_angles, stiffnesses = self.devices[
                i
            ].get_rest_angles_with_stiffnesses()
            tip_end_idx = tip_idx + rest_angles.shape[0]
            rest_angles = (
                rest_angles * stiffnesses / self._stiffnesses[tip_idx:tip_end_idx]
            )
            combined_rest_angles[tip_idx:tip_end_idx] += rest_angles

        rest_angle_changes = np.argwhere(self._last_rest_angles != combined_rest_angles)
        for idx in rest_angle_changes.reshape(-1):
            self._springs[idx].rest_angle = combined_rest_angles[idx]

        self._last_rest_angles = combined_rest_angles
