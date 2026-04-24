from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
from .instrument import (
    Instrument,
    SpireSection,
    StraightMeshElement,
    ArcMeshElement,
    StraightSection,
    MeshSection,
)


@dataclass
class Simmons4Bends(Instrument):
    name: str = "Simmons"
    velocity_limit: Tuple[float, float] = (50, 3.14)

    straights: Tuple[float, float, float, float] = (30, 5, 5, 5)
    radii: Tuple[float, float, float, float] = (10, 6, 20, 20)
    angles_deg: Tuple[float, float, float, float] = (-2, 185, 20, -70)
    diameter_outer: float = 2
    diameter_inner: float = 1
    poisson_ratio: float = 0.49
    young_modulus: float = 1e3
    mass_density: float = 0.000005
    collis_edges_per_mm: float = 2
    visu_edges_per_mm: float = 0.5

    base_length: float = 500
    base_diameter_outer: float = 2
    base_diameter_inner: float = 1
    base_poisson_ratio: float = 0.49
    base_young_modulus: float = 1e3
    base_mass_density: float = 0.000005
    base_collis_edges_per_mm: float = 0.1
    base_visu_edges_per_mm: float = 0.5

    color: Tuple[float, float, float] = (30, 144, 255)

    arc_mesh_resolution: float = 0.1

    tip_section: Union[StraightSection, SpireSection, MeshSection] = field(
        init=False, repr=False, default=None
    )
    base_section: Optional[StraightSection] = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self):

        base_section = StraightSection(
            length=self.base_length,
            diameter_outer=self.base_diameter_outer,
            diameter_inner=self.base_diameter_inner,
            poisson_ratio=self.base_poisson_ratio,
            young_modulus=self.base_young_modulus,
            mass_density=self.base_mass_density,
            collis_edges_per_mm=self.base_collis_edges_per_mm,
            visu_edges_per_mm=self.base_visu_edges_per_mm,
        )

        mesh_elements = []
        for i in range(len(self.straights)):
            if self.angles_deg[i] == 0 or self.radii[i] == 0:
                raise ValueError(
                    f"Neither {self.angles_deg[i]=} and {self.radii[i]=} may be 0!"
                )
            arc = ArcMeshElement(
                self.radii[i], self.angles_deg[i], 0.0, self.arc_mesh_resolution
            )
            mesh_elements.append(arc)

            if self.straights[i] <= 0:
                raise ValueError(f"{self.straights[i]} must be >0!")

            straight = StraightMeshElement(self.straights[i])
            mesh_elements.append(straight)

        tip_section = MeshSection(
            diameter_outer=self.diameter_outer,
            diameter_inner=self.diameter_inner,
            poisson_ratio=self.poisson_ratio,
            young_modulus=self.young_modulus,
            mass_density=self.mass_density,
            visu_edges_per_mm=self.visu_edges_per_mm,
            collis_edges_per_mm=self.collis_edges_per_mm,
            mesh_elements=mesh_elements,
        )

        self.tip_section = tip_section
        self.base_section = base_section


@dataclass
class Simmons3Bends(Simmons4Bends):

    name: str = "Simmons"
    velocity_limit: Tuple[float, float] = (50, 3.14)

    straights: Tuple[float, float, float] = (5, 10, 5)
    radii: Tuple[float, float, float] = (6, 20, 25)
    angles_deg: Tuple[float, float, float] = (185, 20, -60)
