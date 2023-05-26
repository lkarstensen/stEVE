from dataclasses import dataclass
from typing import Tuple
import math
from .sofadevice import ProceduralShape
from .device import Device


@dataclass
class JShaped(Device):
    """_summary_

    Args:
        name (str): name of the device
        velocity_limit: (Tuple[float, float]): Maximum speed for translation in mm/s and rotation in rad/s
        total_length (float): Total length of device
        tip_length (float): Lngeht of tip
        tip_angle (float): Amount of circle the tip does in radians.
        tip_outer_diameter (float): Diameter of the device along its axis at the tip.
        tip_inner_radius (float): Inner diamter of the device along its axis at the tip.
        straight_outer_diameter (float):  Diameter of the device along its axis at the straight part.
        straight_inner_radius (float): Inner diamter of the device along its axis at the straight part.
        poisson_ratio (float): possion ratio of the material for the total length
        young_modulus_tip (float): young modulus for beams of tip
        young_modulus_straight (float): young modulus for beams of straight length
        mass_density_tip (float):  mass density at tip
        mass_density_straight (float): mass density of straight length
        visu_edges_per_mm (float): Density of visualisation edges along the total length
        collis_edges_per_mm_tip (float): Density of collision edges along the tip.
        collis_edges_per_mm_straight (float): Density of collision edges along straight part.
        beams_per_mm_tip (float): Density of FEM beams of the tip.
        beams_per_mm_straight (float): Density of FEM beam of the straight part.
    """

    name: str = "guidewire"
    velocity_limit: Tuple[float, float] = (50, 3.14)
    length: float = 450
    tip_radius: float = 12.1
    tip_angle: float = 0.4 * math.pi
    tip_outer_diameter: float = 0.7
    tip_inner_diameter: float = 0.0
    straight_outer_diameter: float = 0.89
    straight_inner_diameter: float = 0.0
    poisson_ratio: float = 0.49
    young_modulus_tip: float = 17e3
    young_modulus_straight: float = 80e3
    mass_density_tip: float = 0.000021
    mass_density_straight: float = 0.000021
    visu_edges_per_mm: float = 0.5
    collis_edges_per_mm_tip: float = 2
    collis_edges_per_mm_straight: float = 0.1
    beams_per_mm_tip: float = 1.4
    beams_per_mm_straight: float = 0.09
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        spire_height = 0.0
        tip_length = self.tip_radius * self.tip_angle
        straight_length = self.length - tip_length
        spire_diameter = self.tip_radius * 2
        num_edges = math.ceil(self.visu_edges_per_mm * self.length)
        num_edges_collis_tip = math.ceil(self.collis_edges_per_mm_tip * tip_length)
        num_edges_collis_straight = math.ceil(
            self.collis_edges_per_mm_straight * straight_length
        )
        beams_tip = math.ceil(tip_length * self.beams_per_mm_tip)
        beams_straight = math.ceil(straight_length * self.beams_per_mm_straight)

        self.sofa_device = ProceduralShape(
            length=self.length,
            straight_length=straight_length,
            spire_diameter=spire_diameter,
            spire_height=spire_height,
            poisson_ratio=self.poisson_ratio,
            young_modulus=self.young_modulus_straight,
            young_modulus_extremity=self.young_modulus_tip,
            radius=self.straight_outer_diameter / 2,
            radius_extremity=self.tip_outer_diameter / 2,
            inner_radius=self.straight_inner_diameter / 2,
            inner_radius_extremity=self.tip_inner_diameter / 2,
            mass_density=self.mass_density_straight,
            mass_density_extremity=self.mass_density_tip,
            num_edges=num_edges,
            num_edges_collis=(num_edges_collis_straight, num_edges_collis_tip),
            density_of_beams=(beams_straight, beams_tip),
            key_points=(0.0, straight_length, self.length),
            color=self.color,
        )
