from typing import List, Tuple
from .multidevicemp import MutliDeviceMP, VesselTree, Device3D


class CTRMP(MutliDeviceMP):
    def __init__(
        self,
        vessel_tree: VesselTree,
        devices: List[Device3D],
        stop_device_at_tree_end: bool = True,
        control_frequency: float = 30,
        dt_simulation: float = 0.006,
        sofa_native_gui: bool = False,
        step_timeout: float = 10,
        sofa_restart_episodes: float = 25,
        tracking_low_custom: Tuple[float, float, float] = None,
        tracking_high_custom: Tuple[float, float, float] = None,
    ) -> None:
        self.control_frequency = control_frequency
        super().__init__(
            vessel_tree,
            devices,
            stop_device_at_tree_end,
            control_frequency,
            dt_simulation,
            sofa_native_gui,
            step_timeout,
            sofa_restart_episodes,
            tracking_low_custom,
            tracking_high_custom,
        )

    def _add_vessel_tree(self, mesh_path):
        ...
        # self._vessel_object = self._root.addChild("target")
        # self._vessel_object.addObject(
        #     "MeshObjLoader",
        #     name="loader",
        #     triangulate=True,
        #     filename="/Users/lennartkarstensen/stacie/eve/eve/visualisation/meshes/sphere.obj",
        #     scale=20,
        # )
        # self._vessel_object.addObject("MechanicalObject", src="@loader")
        # self._vessel_object.addObject(
        #     "OglModel",
        #     src="@loader",
        #     color=[0.5, 1.0, 1.0, 0.3],
        # )
