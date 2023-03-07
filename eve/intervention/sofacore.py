import importlib
from typing import List, Optional, Tuple
import logging
import numpy as np
from .device import Device


class SOFACore:
    def __init__(
        self,
        devices: List[Device],
        image_frequency: float = 7.5,
        dt_simulation: float = 0.006,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        self.devices = tuple(devices)
        self.image_frequency = image_frequency
        self.dt_simulation = dt_simulation

        self.root = None
        self.camera = None
        self.simulation_error = False

        self._vessel_object = None
        self._instruments_combined = None
        self._device_dofs = []
        self.sofa_initialized = False

        self._sofa = None
        self._sofa_runtime = None

    @property
    def dof_positions(self) -> np.ndarray:
        tracking = self._instruments_combined.DOFs.position.value[:, 0:3][::-1]
        if np.any(np.isnan(tracking[0])):
            self.logger.warning("Tracking is NAN, resetting devices")
            self.simulation_error = True
            self.reset_sofa_devices()
            tracking = self._instruments_combined.DOFs.position.value[:, 0:3][::-1]
        return tracking

    @property
    def inserted_lengths(self) -> List[float]:
        return self._instruments_combined.m_ircontroller.xtip.value

    @property
    def rotations(self) -> List[float]:
        try:
            rots = self._instruments_combined.m_ircontroller.rotationInstrument.value
        except AttributeError:
            rots = [0.0] * len(self.devices)
        return rots

    def unload_simulation(self):
        self._sofa.Simulation.unload(self.root)

    def do_sofa_step(self, action: np.ndarray):

        inserted_lengths = self.inserted_lengths

        if len(self.devices) > 1:
            max_id = np.argmax(inserted_lengths)
            new_length = inserted_lengths + action[:, 0] / self.image_frequency
            new_max_id = np.argmax(new_length)
            if max_id != new_max_id:
                if abs(action[max_id, 0]) > abs(action[new_max_id, 0]):
                    action[new_max_id, 0] = 0.0
                else:
                    action[max_id, 0] = 0.0

        x_tip = self._instruments_combined.m_ircontroller.xtip
        tip_rot = self._instruments_combined.m_ircontroller.rotationInstrument
        for i in range(action.shape[0]):
            x_tip[i] += float(action[i][0] * self.root.dt.value)
            tip_rot[i] += float(action[i][1] * self.root.dt.value)
        self._instruments_combined.m_ircontroller.xtip = x_tip
        self._instruments_combined.m_ircontroller.rotationInstrument = tip_rot
        self._sofa.Simulation.animate(self.root, self.root.dt.value)

    def reset_sofa_devices(self):

        x = self._instruments_combined.m_ircontroller.xtip.value
        self._instruments_combined.m_ircontroller.xtip.value = x * 0.0
        ri = self._instruments_combined.m_ircontroller.rotationInstrument.value
        self._instruments_combined.m_ircontroller.rotationInstrument.value = ri * 0.0
        self._instruments_combined.m_ircontroller.indexFirstNode.value = 0
        self._sofa.Simulation.reset(self.root)

    def init_sofa(
        self,
        insertion_point,
        insertion_direction,
        mesh_path,
        add_visual: bool = False,
        display_size: Optional[Tuple[int, int]] = None,
        coords_high: Optional[Tuple[float, float, float]] = None,
        coords_low: Optional[Tuple[float, float, float]] = None,
    ):
        self._sofa = importlib.import_module("Sofa")
        self._sofa_runtime = importlib.import_module("SofaRuntime")
        if self.sofa_initialized:
            self.unload_simulation()
        if self.root is None:
            self.root = self._sofa.Core.Node()
        self.root.gravity = [0.0, 0.0, 0.0]
        self.root.dt = self.dt_simulation
        self._load_plugins()
        self._basic_setup()
        self._add_vessel_tree(mesh_path=mesh_path)
        self._add_device(
            insertion_point=insertion_point, insertion_direction=insertion_direction
        )
        if add_visual:
            self._add_visual(display_size, coords_low, coords_high)

        self._sofa.Simulation.init(self.root)
        self.sofa_initialized = True
        self.simulation_error = False
        self.logger.info("Sofa Initialized")

    def _load_plugins(self):
        self.root.addObject(
            "RequiredPlugin",
            pluginName="\
            BeamAdapter\
            Sofa.Component.AnimationLoop\
            Sofa.Component.Collision.Detection.Algorithm\
            Sofa.Component.Collision.Detection.Intersection\
            Sofa.Component.LinearSolver.Direct\
            Sofa.Component.IO.Mesh",
        )

    def _basic_setup(self):
        self.root.addObject("FreeMotionAnimationLoop")
        self.root.addObject("DefaultPipeline", draw="0", depth="6", verbose="1")
        self.root.addObject("BruteForceBroadPhase")
        self.root.addObject("BVHNarrowPhase")
        self.root.addObject(
            "LocalMinDistance",
            contactDistance=0.3,
            alarmDistance=0.5,
            angleCone=0.02,
            name="localmindistance",
        )
        self.root.addObject(
            "DefaultContactManager", response="FrictionContactConstraint"
        )
        self.root.addObject(
            "LCPConstraintSolver",
            mu=0.1,
            tolerance=1e-4,
            maxIt=2000,
            name="LCP",
            build_lcp=False,
        )

    def _add_vessel_tree(self, mesh_path):

        vessel_object = self.root.addChild("vesselTree")
        vessel_object.addObject(
            "MeshObjLoader",
            filename=mesh_path,
            flipNormals=False,
            name="meshLoader",
        )
        vessel_object.addObject(
            "MeshTopology",
            position="@meshLoader.position",
            triangles="@meshLoader.triangles",
        )
        vessel_object.addObject("MechanicalObject", src="@meshLoader")
        vessel_object.addObject("TriangleCollisionModel", moving=False, simulated=False)
        vessel_object.addObject("LineCollisionModel", moving=False, simulated=False)
        self._vessel_object = vessel_object

    def _add_device(self, insertion_point, insertion_direction):

        for device in self.devices:
            topo_lines = self.root.addChild("topolines_" + device.name)
            if not device.is_a_procedural_shape:
                topo_lines.addObject(
                    "MeshObjLoader",
                    filename=device.mesh_path,
                    name="loader",
                )
            topo_lines.addObject(
                "WireRestShape",
                name="rest_shape_" + device.name,
                isAProceduralShape=device.is_a_procedural_shape,
                straightLength=device.straight_length,
                length=device.length,
                spireDiameter=device.spire_diameter,
                radiusExtremity=device.radius_extremity,
                youngModulusExtremity=device.young_modulus_extremity,
                massDensityExtremity=device.mass_density_extremity,
                radius=device.radius,
                youngModulus=device.young_modulus,
                massDensity=device.mass_density,
                poissonRatio=device.poisson_ratio,
                keyPoints=device.key_points,
                densityOfBeams=device.density_of_beams,
                numEdgesCollis=device.num_edges_collis,
                numEdges=device.num_edges,
                spireHeight=device.spire_height,
                printLog=True,
                template="Rigid3d",
            )
            topo_lines.addObject(
                "EdgeSetTopologyContainer", name="meshLines_" + device.name
            )
            topo_lines.addObject("EdgeSetTopologyModifier", name="Modifier")
            topo_lines.addObject(
                "EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d"
            )
            topo_lines.addObject(
                "MechanicalObject", name="dofTopo_" + device.name, template="Rigid3d"
            )

        instrument_combined = self.root.addChild("InstrumentCombined")
        instrument_combined.addObject(
            "EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=0.1
        )
        instrument_combined.addObject(
            "BTDLinearSolver", verification=False, subpartSolve=False, verbose=False
        )
        nx = 0
        for device in self.devices:
            nx = sum([nx, sum(device.density_of_beams)])

        instrument_combined.addObject(
            "RegularGridTopology",
            name="MeshLines",
            nx=nx,
            ny=1,
            nz=1,
            xmax=1.0,
            xmin=0.0,
            ymin=0,
            ymax=0,
            zmax=1,
            zmin=1,
            p0=[0, 0, 0],
        )
        instrument_combined.addObject(
            "MechanicalObject",
            showIndices=False,
            name="DOFs",
            template="Rigid3d",
        )
        x_tip = []
        rotations = []
        interpolations = ""

        for device in self.devices:
            wire_rest_shape = (
                "@../topolines_" + device.name + "/rest_shape_" + device.name
            )
            instrument_combined.addObject(
                "WireBeamInterpolation",
                name="Interpol_" + device.name,
                WireRestShape=wire_rest_shape,
                radius=device.radius,
                printLog=False,
            )
            instrument_combined.addObject(
                "AdaptiveBeamForceFieldAndMass",
                name="ForceField_" + device.name,
                massDensity=device.mass_density,
                interpolation="@Interpol_" + device.name,
            )
            x_tip.append(0.0)
            rotations.append(0.0)
            interpolations += "Interpol_" + device.name + " "
        x_tip[0] += 0.1
        interpolations = interpolations[:-1]

        insertion_pose = self._calculate_insertion_pose(
            insertion_point, insertion_direction
        )

        instrument_combined.addObject(
            "InterventionalRadiologyController",
            name="m_ircontroller",
            template="Rigid3d",
            instruments=interpolations,
            startingPos=insertion_pose,
            xtip=x_tip,
            printLog=True,
            rotationInstrument=rotations,
            speed=0.0,
            listening=True,
            controlledInstrument=0,
        )
        instrument_combined.addObject(
            "LinearSolverConstraintCorrection", wire_optimization="true", printLog=False
        )
        instrument_combined.addObject(
            "FixedConstraint", indices=0, name="FixedConstraint"
        )
        instrument_combined.addObject(
            "RestShapeSpringsForceField",
            points="@m_ircontroller.indexFirstNode",
            angularStiffness=1e8,
            stiffness=1e8,
            external_points=0,
            external_rest_shape="@DOFs",
        )
        self._instruments_combined = instrument_combined

        beam_collis = instrument_combined.addChild("CollisionModel")
        beam_collis.activated = True
        beam_collis.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
        beam_collis.addObject("EdgeSetTopologyModifier", name="colliseEdgeModifier")
        beam_collis.addObject("MechanicalObject", name="CollisionDOFs")
        beam_collis.addObject(
            "MultiAdaptiveBeamMapping",
            controller="../m_ircontroller",
            useCurvAbs=True,
            printLog=False,
            name="collisMap",
        )
        beam_collis.addObject("LineCollisionModel", proximity=0.0)
        beam_collis.addObject("PointCollisionModel", proximity=0.0)

    def _add_visual(
        self,
        display_size: Tuple[int, int],
        coords_low: Tuple[float, float, float],
        coords_high: Tuple[float, float, float],
    ):
        coords_low = np.array(coords_low)
        coords_high = np.array(coords_high)
        self.root.addObject(
            "RequiredPlugin",
            pluginName="\
            Sofa.GL.Component.Rendering3D\
            Sofa.GL.Component.Shader",
        )

        # Vessel Tree
        self._vessel_object.addObject(
            "OglModel",
            src="@meshLoader",
            color=[0.5, 1.0, 1.0, 0.3],
        )

        # Devices
        for device in self.devices:
            visu_node = self._instruments_combined.addChild("Visu_" + device.name)
            visu_node.activated = True
            visu_node.addObject("MechanicalObject", name="Quads")
            visu_node.addObject(
                "QuadSetTopologyContainer", name="Container_" + device.name
            )
            visu_node.addObject("QuadSetTopologyModifier", name="Modifier")
            visu_node.addObject(
                "QuadSetGeometryAlgorithms",
                name="GeomAlgo",
                template="Vec3d",
            )
            mesh_lines = "@../../topolines_" + device.name + "/meshLines_" + device.name
            visu_node.addObject(
                "Edge2QuadTopologicalMapping",
                nbPointsOnEachCircle=10,
                radius=device.radius,
                flipNormals="true",
                input=mesh_lines,
                output="@Container_" + device.name,
            )
            visu_node.addObject(
                "AdaptiveBeamMapping",
                interpolation="@../Interpol_" + device.name,
                name="VisuMap_" + device.name,
                output="@Quads",
                isMechanical="false",
                input="@../DOFs",
                useCurvAbs="1",
                printLog="0",
            )
            visu_ogl = visu_node.addChild("VisuOgl")
            visu_ogl.activated = True
            visu_ogl.addObject(
                "OglModel",
                color=device.color,
                quads="@../Container_" + device.name + ".quads",
                material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
                name="Visual",
            )
            visu_ogl.addObject(
                "IdentityMapping",
                input="@../Quads",
                output="@Visual",
            )

        # Camera
        self.root.addObject("DefaultVisualManagerLoop")
        self.root.addObject(
            "VisualStyle",
            displayFlags="showVisualModels\
                hideBehaviorModels\
                hideCollisionModels\
                hideWireframe\
                hideMappings\
                hideForceFields",
        )
        self.root.addObject("LightManager")
        self.root.addObject("DirectionalLight", direction=[0, -1, 0])
        self.root.addObject("DirectionalLight", direction=[0, 1, 0])

        # TODO: Find out how to manipulate background. BackgroundSetting doesn't seem to work
        # self.root.addObject("BackgroundSetting", color=(0.5, 0.5, 0.5, 1.0))

        look_at = (coords_high + coords_low) * 0.5
        distance_coefficient = 1.5
        distance = np.linalg.norm(look_at - coords_low) * distance_coefficient
        position = look_at + np.array([0.0, -distance, 0.0])
        scene_radius = np.linalg.norm(coords_high - coords_low)
        dist_cam_to_center = np.linalg.norm(position - look_at)
        z_clipping_coeff = 5
        z_near_coeff = 0.01
        z_near = dist_cam_to_center - scene_radius
        z_far = (z_near + 2 * scene_radius) * 2
        z_near = z_near * z_near_coeff
        z_min = z_near_coeff * z_clipping_coeff * scene_radius
        if z_near < z_min:
            z_near = z_min
        field_of_view = 70
        look_at = np.array(look_at)
        position = np.array(position)

        self.camera = self.root.addObject(
            "Camera",
            name="camera",
            lookAt=look_at,
            position=position,
            fieldOfView=field_of_view,
            widthViewport=display_size[0],
            heightViewport=display_size[1],
            zNear=z_near,
            zFar=z_far,
            fixedLookAt=False,
        )

    @staticmethod
    def _calculate_insertion_pose(
        insertion_point: np.ndarray, insertion_direction: np.ndarray
    ):

        insertion_direction = insertion_direction / np.linalg.norm(insertion_direction)
        original_direction = np.array([1.0, 0.0, 0.0])
        if np.all(insertion_direction == original_direction):
            w0 = 1.0
            xyz0 = [0.0, 0.0, 0.0]
        elif np.all(np.cross(insertion_direction, original_direction) == 0):
            w0 = 0.0
            xyz0 = [0.0, 1.0, 0.0]
        else:
            half = (original_direction + insertion_direction) / np.linalg.norm(
                original_direction + insertion_direction
            )
            w0 = np.dot(original_direction, half)
            xyz0 = np.cross(original_direction, half)
        xyz0 = list(xyz0)
        pose = list(insertion_point) + list(xyz0) + [w0]
        return pose
