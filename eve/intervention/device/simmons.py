from dataclasses import dataclass
from typing import Tuple
from .device import MeshDevice, StraightPart, Arc


class Simmons4Bends(MeshDevice):
    name: str = "Simmons"
    velocity_limit: Tuple[float, float] = (50, 3.14)
    straights: Tuple[float, float, float, float, float] = (500, 30, 5, 5, 5)
    radii: Tuple[float, float, float, float] = (10, 6, 20, 20)
    angles_deg: Tuple[float, float, float, float] = (-2, 185, 20, -70)
    outer_diameter: float = 2
    inner_diameter: float = 1
    poisson_ratio: float = 0.49
    young_modulus: float = 1e3
    mass_density: float = 0.000005
    visu_edges_per_mm: float = 0.5
    collis_edges_per_mm_tip: float = 2
    collis_edges_per_mm_straight: float = 0.1
    beams_per_mm_tip: float = 1.4
    beams_per_mm_straight: float = 0.09
    color: Tuple[float, float, float] = (30, 144, 255)

    @property
    def length(self) -> float:
        return self.sofa_device.length

    def __post_init__(self):
        elements = []
        elements.append(
            StraightPart(
                self.straights[0],
                self.visu_edges_per_mm,
                self.collis_edges_per_mm_straight,
                self.beams_per_mm_straight,
            )
        )
        for i in range(4):
            if self.angles_deg[i] != 0 and self.radii[i] > 0:
                arc = Arc(
                    self.radii[i],
                    self.angles_deg[i],
                    0.0,
                    self.visu_edges_per_mm,
                    self.collis_edges_per_mm_tip,
                    self.beams_per_mm_tip,
                )
                elements.append(arc)
            if self.straights[i + 1] > 0:
                straight = StraightPart(
                    self.straights[i + 1],
                    self.visu_edges_per_mm,
                    self.collis_edges_per_mm_tip,
                    self.beams_per_mm_tip,
                )
                elements.append(straight)
        super().__init__(
            elements,
            self.outer_diameter,
            self.inner_diameter,
            self.poisson_ratio,
            self.young_modulus,
            self.mass_density,
            self.color,
        )


@dataclass
class Simmons3Bends(Simmons4Bends):
    name: str = "Simmons"
    velocity_limit: Tuple[float, float] = (50, 3.14)
    straights: Tuple[float, float, float, float] = (530, 5, 10, 5)
    radii: Tuple[float, float, float] = (6, 20, 25)
    angles_deg: Tuple[float, float, float] = (185, 20, -60)
    outer_diameter: float = 2
    inner_diameter: float = 1
    poisson_ratio: float = 0.49
    young_modulus: float = 1000
    mass_density: float = 0.000005
    visu_edges_per_mm: float = 0.5
    collis_edges_per_mm_tip: float = 2
    collis_edges_per_mm_straight: float = 0.1
    beams_per_mm_tip: float = 1.4
    beams_per_mm_straight: float = 0.09
    color: Tuple[float, float, float] = (30, 144, 255)

    def __post_init__(self):
        straights = [self.straights[0]] + [0] + list(self.straights)[1:]
        radii = [0] + list(self.radii)
        angles_deg = [0] + list(self.angles_deg)
        object.__setattr__(self, "straights", tuple(straights))
        object.__setattr__(self, "radii", tuple(radii))
        object.__setattr__(self, "angles_deg", tuple(angles_deg))
        super().__post_init__()
