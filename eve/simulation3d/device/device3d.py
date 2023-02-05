from dataclasses import dataclass, field
from typing import Tuple, Union

from ...intervention import Device


@dataclass(frozen=True, init=False)
class Device3D(Device):
    """A container for alle the data necessary to create a WireRestShape in SOFA BeamAdapter

    Args:
        name (str): name of the device
        velocity_limit: (Tuple[float, float]): Maximum speed for translation in mm/s and rotation in rad/s
        is_a_procedural_shape (bool): rest shape is mathematically defined if it is a procedural shape
        mesh_path (str): path to mesh for non procedural shapes
        length (float): Total length
        straight_length (float): straight length (spire_length = length - straight_length) spire = tip
        spire_diameter (float): diameter of the spire
        spire_height (float): height between each spire
        poisson_ratio (float): possion ratio of the material for the total length
        young_modulus (float): young modulus for beams of straight length
        young_modulus_extremity (float): young modulus for beams of spire/tip
        radius (float): radius of straight length
        radius_extremity (float): radius at tip/spire
        inner_radius (float): inner radius of straight length (can be 0)
        inner_radius_extremity (float): inner radius of tip/spire (can be 0)
        mass_density (float): mass density of straight length
        mass_density_extremity (float): mass density at tip/spire
        num_edges (int): number of Edges for the visual model
        num_edges_collis (Union[int, List[int]]): number of Edges for the collision model between key points. len(num_edges_collis) has to be len(key_points) - 1
        density_of_beams (Union[int, List[int]]): number of beams between key points. Naming density_of_beams is confusing, but is copied from BeamAdapter. len(density_of_beams) has to be len(key_points) - 1
        key_points (List[float]): key points for beam density. Normally [0.0, straight_length, length] or [0.0, length] if straight_length == length
        color (Tuple[float, float, float]): color as [R,G,B]
    """

    name: str
    velocity_limit: Tuple[float, float]
    is_a_procedural_shape: bool
    mesh_path: str
    length: float
    straight_length: float
    spire_diameter: float
    spire_height: float
    poisson_ratio: float
    young_modulus: float
    young_modulus_extremity: float
    radius: float
    radius_extremity: float
    inner_radius: float
    inner_radius_extremity: float
    mass_density: float
    mass_density_extremity: float
    num_edges: int
    num_edges_collis: Union[int, Tuple[int]]
    density_of_beams: Union[int, Tuple[int]]
    key_points: Tuple[float]
    color: Tuple[float, float, float]


@dataclass(frozen=True)
class ProceduralShape(Device3D):
    """A container for alle the data necessary to create a procedural WireRestShape in SOFA BeamAdapter

    Args:
        name (str): name of the device
        velocity_limit: (Tuple[float, float]): Maximum speed for translation in mm/s and rotation in rad/s
        length (float): Total length
        straight_length (float): straight length (spire_length = length - straight_length) spire = tip
        spire_diameter (float): diameter of the spire
        spire_height (float): height between each spire
        poisson_ratio (float): possion ratio of the material for the total length
        young_modulus (float): young modulus for beams of straight length
        young_modulus_extremity (float): young modulus for beams of spire/tip
        radius (float): radius of straight length
        radius_extremity (float): radius at tip/spire
        inner_radius (float): inner radius of straight length (can be 0)
        inner_radius_extremity (float): inner radius of tip/spire (can be 0)
        mass_density (float): mass density of straight length
        mass_density_extremity (float): mass density at tip/spire
        num_edges (int): number of Edges for the visual model
        num_edges_collis (Union[int, List[int]]): number of Edges for the collision model between key points. len(num_edges_collis) has to be len(key_points) - 1
        density_of_beams (Union[int, List[int]]): number of beams between key points. Naming density_of_beams is confusing, but is copied from BeamAdapter. len(density_of_beams) has to be len(key_points) - 1
        key_points (List[float]): key points for beam density. Normally [0.0, straight_length, length] or [0.0, length] if straight_length == length
        color (Tuple[float, float, float]): color as [R,G,B]
    """

    is_a_procedural_shape: bool = field(init=False, repr=False, default=True)
    mesh_path: str = field(init=False, repr=False, default=None)


@dataclass(frozen=True)
class NonProceduralShape(Device3D):
    """A container for alle the data necessary to create a non-procedural WireRestShape in SOFA BeamAdapter

    Args:
        name (str): name of the device
        velocity_limit: (Tuple[float, float]): Maximum speed for translation in mm/s and rotation in rad/s
        mesh_path (str): _description_
        length (float): Total length
        poisson_ratio (float): possion ratio of the material for the total length
        young_modulus (float): young modulus for beams of straight length
        radius (float): radius of straight length
        inner_radius (float): inner radius of straight length (can be 0)
        mass_density (float): mass density of straight length
        num_edges (int): number of Edges for the visual model
        num_edges_collis (Union[int, List[int]]): number of Edges for the collision model between key points. len(num_edges_collis) has to be len(key_points) - 1
        density_of_beams (Union[int, List[int]]): number of beams between key points. Naming density_of_beams is confusing, but is copied from BeamAdapter. len(density_of_beams) has to be len(key_points) - 1
        key_points (List[float]): key points for beam density. Normally [0.0, straight_length, length] or [0.0, length] if straight_length == length
        color (Tuple[float, float, float]): color as [R,G,B]
    """

    is_a_procedural_shape: bool = field(init=False, repr=False, default=False)
    straight_length: float = field(init=False, repr=False, default=0.0)
    spire_diameter: float = field(init=False, repr=False, default=0.0)
    spire_height: float = field(init=False, repr=False, default=0.0)
    young_modulus_extremity: float = field(init=False, repr=False, default=0.0)
    radius_extremity: float = field(init=False, repr=False, default=0.0)
    inner_radius_extremity: float = field(init=False, repr=False, default=0.0)
    mass_density_extremity: float = field(init=False, repr=False, default=0.0)
