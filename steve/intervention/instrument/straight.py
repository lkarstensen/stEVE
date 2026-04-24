from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

from .instrument import (
    Instrument,
    StraightSection,
    MeshSection,
    StraightMeshElement,
    SpireSection,
)


@dataclass
class Straight(Instrument):

    name: str = "guidewire"
    velocity_limit: Tuple[float, float] = (50, 3.14)

    length: float = 450
    diameter_outer: float = 0.89
    diameter_inner: float = 0.0
    young_modulus: float = 80e3
    mass_density: float = 0.000021
    poisson_ratio: float = 0.49
    collis_edges_per_mm: float = 0.1
    visu_edges_per_mm: float = 0.5

    flex_length: float = 30.0
    flex_diameter_outer: float = 0.7
    flex_diameter_inner: float = 0.0
    flex_young_modulus: float = 17e3
    flex_mass_density: float = 0.000021
    flex_poisson_ratio: float = 0.49
    flex_collis_edges_per_mm: float = 2
    flex_visu_edges_per_mm: float = 0.5
    flex_length: float = 30

    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    tip_section: Union[StraightSection, SpireSection, MeshSection] = field(
        init=False, repr=False, default=None
    )
    base_section: Optional[StraightSection] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):
        stiff_section = StraightSection(
            length=self.length - self.flex_length,
            diameter_outer=self.diameter_outer,
            diameter_inner=self.diameter_inner,
            poisson_ratio=self.poisson_ratio,
            young_modulus=self.young_modulus,
            mass_density=self.mass_density,
            visu_edges_per_mm=self.visu_edges_per_mm,
            collis_edges_per_mm=self.collis_edges_per_mm,
        )

        flex_straight = StraightMeshElement(self.flex_length)

        flex_section = MeshSection(
            diameter_outer=self.flex_diameter_outer,
            diameter_inner=self.flex_diameter_inner,
            poisson_ratio=self.flex_poisson_ratio,
            young_modulus=self.flex_young_modulus,
            mass_density=self.flex_mass_density,
            visu_edges_per_mm=self.flex_visu_edges_per_mm,
            collis_edges_per_mm=self.flex_collis_edges_per_mm,
            mesh_elements=[flex_straight],
        )

        self.tip_section = flex_section
        self.base_section = stiff_section
