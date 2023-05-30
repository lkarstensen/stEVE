from math import cos, sin
from typing import Tuple
import importlib
import numpy as np

from .visualisation import Visualisation
from ..intervention import Intervention


class SofaPygame(Visualisation):
    def __init__(
        self,
        intervention: Intervention,
        display_size: Tuple[float, float] = (1200, 860),
    ) -> None:
        self.intervention = intervention
        self.display_size = display_size
        simulation = self.intervention.simulation
        simulation.init_visual_nodes = True
        simulation.display_size = display_size
        simulation.target_size = self.intervention.target.threshold

        self.initial_orientation = None
        self._initialized = False
        self._theta_x = intervention.fluoroscopy.image_rot_zx[1] * np.pi / 180
        self._theta_z = intervention.fluoroscopy.image_rot_zx[0] * np.pi / 180
        self._initial_direction = None
        self._initial_look_at = None
        self._distance = None
        self._sofa = None
        self._sofa_gl = None
        self._opengl_gl = None
        self._opengl_glu = None
        self._pygame = None

    def render(self) -> None:
        simulation = self.intervention.simulation
        self._sofa.Simulation.updateVisual(simulation.root)
        gl = self._opengl_gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        camera = simulation.camera
        width = camera.widthViewport.value
        height = camera.heightViewport.value
        self._opengl_glu.gluPerspective(
            camera.fieldOfView.value,
            (width / height),
            camera.zNear.value,
            camera.zFar.value,
        )
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        camera_mvm = camera.getOpenGLModelViewMatrix()
        gl.glMultMatrixd(camera_mvm)
        self._sofa_gl.draw(simulation.root)
        gl = self._opengl_gl
        height = camera.heightViewport.value
        width = camera.widthViewport.value

        buffer = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image_array = np.fromstring(buffer, np.uint8)

        if image_array != []:
            image = image_array.reshape(height, width, 3)
            image = np.flipud(image)[:, :, :3]
        else:
            image = np.zeros((height, width, 3))
        self._pygame.display.flip()
        return np.copy(image)

    def reset(self, episode_nr: int = 0) -> None:
        simulation = self.intervention.simulation
        # pylint: disable=no-member
        self._sofa = self._sofa or importlib.import_module("Sofa")
        self._sofa_gl = self._sofa_gl or importlib.import_module("Sofa.SofaGL")
        self._opengl_gl = self._opengl_gl or importlib.import_module("OpenGL.GL")
        self._opengl_glu = self._opengl_glu or importlib.import_module("OpenGL.GLU")
        self._pygame = self._pygame or importlib.import_module("pygame")
        if not self._initialized:
            self._pygame.display.init()
            flags = (
                self._pygame.DOUBLEBUF | self._pygame.OPENGL | self._pygame.RESIZABLE
            )
            self._pygame.display.set_mode(self.display_size, flags)
            self._initialized = True

        gl = self._opengl_gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        self._sofa.SofaGL.glewInit()
        self._sofa.Simulation.initVisual(simulation.root)
        self._sofa.Simulation.initTextures(simulation.root)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        camera = simulation.camera
        width = camera.widthViewport.value
        height = camera.heightViewport.value
        self._opengl_glu.gluPerspective(
            camera.fieldOfView.value,
            (width / height),
            camera.zNear.value,
            camera.zFar.value,
        )
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        if self.initial_orientation is not None:
            self._theta_x = 0.0
            self._theta_z = 0.0
            self.rotate(0, 0)
        self.initial_orientation = np.array(
            [
                camera.orientation[3],
                camera.orientation[0],
                camera.orientation[1],
                camera.orientation[2],
            ]
        )
        position = camera.position
        look_at = camera.lookAt
        self._initial_direction = position - look_at
        self._distance = np.linalg.norm(self._initial_direction)
        self._initial_direction = self._initial_direction / self._distance
        fluoroscopy = self.intervention.fluoroscopy
        self._initial_look_at = fluoroscopy.image_center
        self._theta_x = fluoroscopy.image_rot_zx[1] * np.pi / 180
        self._theta_z = fluoroscopy.image_rot_zx[0] * np.pi / 180
        camera.lookAt = self._initial_look_at
        self.rotate(0, 0)
        target = self.intervention.target.coordinates3d
        target = fluoroscopy.tracking3d_to_vessel_cs(target)
        simulation.target_node.MechanicalObject.translation = [
            target[0],
            target[1],
            target[2],
        ]
        self._sofa.Simulation.init(simulation.target_node)

    def close(self):
        self._pygame.quit()  # pylint: disable=no-member

    def translate(self, velocity: np.array):
        simulation = self.intervention.simulation
        dt = simulation.root.dt.value
        camera = simulation.camera

        position = camera.position
        position += velocity * dt
        camera.position = position

        look_at = camera.lookAt
        look_at += velocity * dt
        camera.lookAt = look_at

    def zoom(self, velocity: float):
        simulation = self.intervention.simulation
        dt = simulation.root.dt.value
        camera = simulation.camera

        position = camera.position
        look_at = camera.lookAt
        direction = look_at - position
        direction = (
            direction
            / (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** 0.5
        )
        position += direction * velocity * dt
        self._distance -= velocity * dt
        camera.position = position

    def rotate(self, lao_rao_speed: float, cra_cau_speed: float):
        simulation = self.intervention.simulation
        dt = simulation.root.dt.value
        camera = simulation.camera

        look_at = camera.lookAt
        self._theta_x += cra_cau_speed * dt
        self._theta_z += lao_rao_speed * dt
        theta_x = self._theta_x
        theta_z = self._theta_z

        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, cos(theta_x), -sin(theta_x)],
                [0, sin(theta_x), cos(theta_x)],
            ]
        )
        rotation_z = np.array(
            [
                [cos(theta_z), -sin(theta_z), 0],
                [sin(theta_z), cos(theta_z), 0],
                [0, 0, 1],
            ]
        )
        rotation = np.matmul(rotation_z, rotation_x)
        offset = np.matmul(rotation, self._initial_direction * self._distance)

        camera.position = look_at + np.array(offset)

        camera_rot_x = np.array([cos(theta_x / 2), sin(theta_x / 2), 0, 0])
        camera_rot_z = np.array([cos(theta_z / 2), 0, 0, sin(theta_z / 2)])

        camera_orientation = self._quat_mult(camera_rot_x, self.initial_orientation)
        camera_orientation = self._quat_mult(camera_rot_z, camera_orientation)

        camera.orientation = np.array(
            [
                camera_orientation[1],
                camera_orientation[2],
                camera_orientation[3],
                camera_orientation[0],
            ]
        )

    @staticmethod
    def _quat_mult(x, y):
        return np.array(
            [
                x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3],
                x[0] * y[1] + x[1] * y[0] + x[2] * y[3] - x[3] * y[2],
                x[0] * y[2] - x[1] * y[3] + x[2] * y[0] + x[3] * y[1],
                x[0] * y[3] + x[1] * y[2] - x[2] * y[1] + x[3] * y[0],
            ]
        )
