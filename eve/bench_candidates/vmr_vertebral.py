# pylint: disable=no-member

from time import perf_counter
import pygame
import numpy as np
import eve
import eve.visualisation


class VertebralStaticEnv(eve.Env):
    def __init__(self, visualisation: bool = False) -> None:
        vessel_tree = eve.vesseltree.VMR(
            "/Users/lennartkarstensen/stacie/eve_toolbox/eve/vesseltree/data/vmr/0166_0001",
            5,
            2,
            insertion_vessel_name="lva",
            rotate_yzx_deg=[0, 180, 0],
        )
        device = eve.intervention.device.JWire(
            tip_outer_diameter=0.3,
            straight_outer_diameter=0.3,
            beams_per_mm_straight=0.9,
        )
        device2 = eve.intervention.device.JWire(
            name="cath",
            tip_outer_diameter=0.4,
            straight_outer_diameter=0.4,
            tip_inner_diameter=0.32,
            straight_inner_diameter=0.32,
            color=(1.0, 0.0, 0.0),
        )

        simulation = eve.intervention.Intervention(
            vessel_tree=vessel_tree,
            devices=[device, device2],
            lao_rao_deg=-5,
            cra_cau_deg=20,
        )
        start = eve.start.MaxDeviceLength(simulation, 380)
        target = eve.target.CenterlineRandom(vessel_tree, simulation, threshold=10)
        success = eve.success.TargetReached(target)
        pathfinder = eve.pathfinder.BruteForceBFS(vessel_tree, simulation, target)

        tracking = eve.observation.Tracking(simulation, n_points=5)
        tracking = eve.observation.wrapper.RelativeToFirstRow(tracking)

        target_state = eve.observation.Target(target)
        target_state = eve.observation.wrapper.ToTrackingCS(
            target_state, intervention=simulation
        )

        rotation = eve.observation.Rotations(simulation)
        state = eve.observation.ObsDict(
            {"tracking": tracking, "target": target_state, "rotation": rotation}
        )

        target_reward = eve.reward.TargetReached(target, factor=1.0)
        # step_reward = eve.reward.Step(factor=-0.01)
        path_delta = eve.reward.PathLengthDelta(pathfinder, 0.01)
        reward = eve.reward.Combination([target_reward, path_delta])

        terminal = eve.terminal.TargetReached(target=target)
        truncation = eve.truncation.MaxSteps(200)
        if visualisation:
            visu = eve.visualisation.SofaPygame(simulation)
        else:
            visu = None
        super().__init__(
            vessel_tree,
            simulation,
            target,
            start,
            success,
            state,
            reward,
            terminal,
            truncation=truncation,
            pathfinder=pathfinder,
            visualisation=visu,
        )


if __name__ == "__main__":
    env = VertebralStaticEnv(True)

    r_cum = 0.0

    env.reset()
    last_tracking = None
    while True:
        start = perf_counter()
        trans = 0.0
        rot = 0.0
        camera_trans = np.array([0.0, 0.0, 0.0])
        camera_rot = np.array([0.0, 0.0, 0.0])
        zoom = 0
        pygame.event.get()
        keys_pressed = pygame.key.get_pressed()

        if keys_pressed[pygame.K_ESCAPE]:
            break
        if keys_pressed[pygame.K_UP]:
            trans += 25
        if keys_pressed[pygame.K_DOWN]:
            trans -= 25
        if keys_pressed[pygame.K_LEFT]:
            rot += 1 * 3.14
        if keys_pressed[pygame.K_RIGHT]:
            rot -= 1 * 3.14
        if keys_pressed[pygame.K_r]:
            lao_rao = 0
            cra_cau = 0
            if keys_pressed[pygame.K_d]:
                lao_rao += 10
            if keys_pressed[pygame.K_a]:
                lao_rao -= 10
            if keys_pressed[pygame.K_w]:
                cra_cau -= 10
            if keys_pressed[pygame.K_s]:
                cra_cau += 10
            env.visualisation.rotate(lao_rao, cra_cau)
        else:
            if keys_pressed[pygame.K_w]:
                camera_trans += np.array([0.0, 0.0, 200.0])
            if keys_pressed[pygame.K_s]:
                camera_trans -= np.array([0.0, 0.0, 200.0])
            if keys_pressed[pygame.K_a]:
                camera_trans -= np.array([200.0, 0.0, 0.0])
            if keys_pressed[pygame.K_d]:
                camera_trans = np.array([200.0, 0.0, 0.0])
            env.visualisation.translate(camera_trans)
        if keys_pressed[pygame.K_e]:
            env.visualisation.zoom(1000)
        if keys_pressed[pygame.K_q]:
            env.visualisation.zoom(-1000)

        if keys_pressed[pygame.K_v]:
            action = ((0, 0), (trans, rot))

        else:
            action = ((trans, rot), (0, 0))
        s, r, d, i, success = env.step(action=action)
        env.render()

        if keys_pressed[pygame.K_RETURN]:
            env.reset()
            n_steps = 0

    env.close()
