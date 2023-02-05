from dataclasses import dataclass, field
from typing import List, Tuple, Union
import math
from .device3d import ProceduralShape


@dataclass(frozen=True)
class CTRTube(ProceduralShape):
    """CTR Tube for CTR Simulation

    Args:
        name (str): name of the device
        velocity_limit: (Tuple[float, float]): Maximum speed for translation in mm/s and rotation in rad/s.
        total_length (float): Total length of device.
        tip_radius (float): Preformed Radius of tip.
        tip_angle (float): Amount of circle the tip does in radians.
        outer_diameter (float, optional): Diameter of the device along its axis. Defaults to 0.7.
        inner_diameter (float, optional): Inner diameter of the device along its axis at the tip. Defaults to 0.0.
        poisson_ratio (float, optional): possion ratio of the material for the total length. Defaults to 0.49.
        young_modulus (float, optional): young modulus of the device. Defaults to 1e10.
        mass_density (float, optional): mass density of the device. Defaults to 1e-7.
        visu_edges_per_mm (float, optional): Density of visualisation edges along the total length. Defaults to 0.5.
        collis_edges_per_mm_tip (float, optional): Density of collision edges along the tip.. Defaults to 2.
        collis_edges_per_mm_straight (float, optional): Density of collision edges along straight part. Defaults to 0.1.
        beams_per_mm_tip (float, optional): Density of FEM beams of the tip. Defaults to 1.4.
        beams_per_mm_straight (float, optional): Density of FEM beam of the straight part.. Defaults to 0.09.
        color (Tuple[float, float, float], optional): [R,G,B] color in SOFA. Defaults to [1.0, 0.0, 0.0].
    """

    name: str = ("inner_tube",)
    velocity_limit: Tuple[float, float] = (40, 3.14)
    length: float = 250
    tip_radius: float = 50
    tip_angle: float = 0.7 * math.pi
    outer_diameter: float = 0.7
    inner_diameter: float = 0.0
    poisson_ratio: float = 0.49
    young_modulus: float = 1e10
    mass_density: float = 1e-7
    visu_edges_per_mm: float = 0.5
    collis_edges_per_mm_tip: float = 2
    collis_edges_per_mm_straight: float = 0.1
    beams_per_mm_tip: float = 1.4
    beams_per_mm_straight: float = 0.09
    color: Tuple[float, float, float] = 1.0, 0.0, 0.0

    straight_length: float = field(init=False, repr=False, default=0.0)
    spire_diameter: float = field(init=False, repr=False, default=0.0)
    spire_height: float = field(init=False, repr=False, default=0.0)
    young_modulus: float = field(init=False, repr=False, default=0.0)
    young_modulus_extremity: float = field(init=False, repr=False, default=0.0)
    radius: float = field(init=False, repr=False, default=0.0)
    radius_extremity: float = field(init=False, repr=False, default=0.0)
    inner_radius: float = field(init=False, repr=False, default=0.0)
    inner_radius_extremity: float = field(init=False, repr=False, default=0.0)
    mass_density: float = field(init=False, repr=False, default=0.0)
    mass_density_extremity: float = field(init=False, repr=False, default=0.0)
    num_edges: int = field(init=False, repr=False, default=0)
    num_edges_collis: Union[int, List[int]] = field(init=False, repr=False, default=0)
    density_of_beams: Union[int, List[int]] = field(init=False, repr=False, default=0)
    key_points: List[float] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        tip_outer_diameter = self.outer_diameter
        tip_inner_diameter = self.inner_diameter
        straight_outer_diameter = self.outer_diameter
        straight_inner_diameter = self.inner_diameter
        young_modulus_straight = self.young_modulus
        young_modulus_tip = self.young_modulus
        mass_density_straight = self.mass_density
        mass_density_tip = self.mass_density

        spire_height = 0.0
        spire_diameter = self.tip_radius * 2
        tip_length = spire_diameter * self.tip_angle / 2
        straight_length = self.length - tip_length
        num_edges = math.ceil(self.visu_edges_per_mm * self.length)
        num_edges_collis_tip = math.ceil(self.collis_edges_per_mm_tip * tip_length)
        num_edges_collis_straight = math.ceil(
            self.collis_edges_per_mm_straight * straight_length
        )
        beams_tip = math.ceil(tip_length * self.beams_per_mm_tip)
        beams_straight = math.ceil(straight_length * self.beams_per_mm_straight)

        super().__init__(
            name=self.name,
            velocity_limit=self.velocity_limit,
            length=self.length,
            straight_length=straight_length,
            spire_diameter=spire_diameter,
            spire_height=spire_height,
            poisson_ratio=self.poisson_ratio,
            young_modulus=young_modulus_straight,
            young_modulus_extremity=young_modulus_tip,
            radius=straight_outer_diameter / 2,
            radius_extremity=tip_outer_diameter / 2,
            inner_radius=straight_inner_diameter / 2,
            inner_radius_extremity=tip_inner_diameter / 2,
            mass_density=mass_density_straight,
            mass_density_extremity=mass_density_tip,
            num_edges=num_edges,
            num_edges_collis=[num_edges_collis_straight, num_edges_collis_tip],
            density_of_beams=[beams_straight, beams_tip],
            key_points=[0.0, straight_length, self.length],
            color=self.color,
        )
