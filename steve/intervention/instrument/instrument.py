from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from ...util import EveObject
from .util.meshing import (
    create_shape_point_cloud,
    save_line_mesh,
    StraightMeshElement,
    ArcMeshElement,
)


@dataclass
class InstrumentSection:
    length: float
    diameter_outer: float
    diameter_inner: float
    poisson_ratio: float
    young_modulus: float
    mass_density: float
    visu_edges_per_mm: float
    collis_edges_per_mm: float

    visu_edges: int = field(init=False, repr=False, default=None)
    collis_edges: int = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.visu_edges = int(np.ceil(self.length * self.visu_edges_per_mm))
        self.collis_edges = int(np.ceil(self.length * self.collis_edges_per_mm))


@dataclass
class StraightSection(InstrumentSection): ...


@dataclass
class SpireSection(InstrumentSection):
    spire_radius: float
    spire_height: float
    spire_angle_deg: float

    length: float = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # https://www.redcrab-software.com/de/Rechner/Geometrie/Helix
        r = self.spire_radius
        h = self.spire_height
        slope = h / (2 * np.pi * r)
        curve = 1 / (r * (1 + slope**2))
        length = 2 * np.pi * r * np.sqrt(1 + curve**2) * self.spire_angle_deg / 360
        self.length = length.tolist()

        InstrumentSection.__post_init__(self)


@dataclass
class MeshSection(InstrumentSection):
    mesh_elements: List[Union[StraightMeshElement, ArcMeshElement]]

    length: float = field(init=False, repr=False, default=None)
    mesh_path: str = field(init=False, repr=False, default=None)

    def __post_init__(self):
        point_cloud, length = create_shape_point_cloud(self.mesh_elements)
        self.mesh_path = save_line_mesh(point_cloud)
        self.length = length
        InstrumentSection.__post_init__(self)


@dataclass
class Instrument(EveObject):

    name: str
    velocity_limit: Tuple[float, float]
    tip_section: Union[StraightSection, SpireSection, MeshSection]
    base_section: Optional[StraightSection] = None

    @property
    def length(self) -> float:
        if self.base_section is not None:
            return self.base_section.length + self.tip_section.length

        return self.tip_section.length
