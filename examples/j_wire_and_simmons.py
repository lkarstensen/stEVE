# pylint: disable=no-member

from time import perf_counter
import pygame
import numpy as np
import eve
import eve.visualisation

vessel_tree = eve.intervention.vesseltree.AorticArch()
device = eve.intervention.device.JShaped(beams_per_mm_straight=0.5)
device2 = eve.intervention.device.Simmons3Bends()

simulation = eve.intervention.simulation.Simulation()

fluoroscopy = eve.intervention.fluoroscopy.Fluoroscopy(
    simulation=simulation,
    vessel_tree=vessel_tree,
    image_frequency=7.5,
    image_rot_zx=[20, 5],
)

target = eve.intervention.target.CenterlineRandom(
    vessel_tree=vessel_tree,
    fluoroscopy=fluoroscopy,
    threshold=5,
    branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
)


intervention = eve.intervention.MonoPlaneStatic(
    vessel_tree=vessel_tree,
    devices=[device, device2],
    simulation=simulation,
    fluoroscopy=fluoroscopy,
    target=target,
)


start = eve.start.MaxDeviceLength(intervention=intervention, max_length=500)
pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)


tracking = eve.observation.Tracking2D(intervention, n_points=5)
tracking = eve.observation.wrapper.RelativeToFirstRow(tracking)

target_state = eve.observation.Target2D(intervention)

rotation = eve.observation.Rotations(intervention)
state = eve.observation.ObsDict(
    {"tracking": tracking, "target": target_state, "rotation": rotation}
)

target_reward = eve.reward.TargetReached(intervention, factor=1.0)
# step_reward = eve.reward.Step(factor=-0.01)
path_delta = eve.reward.PathLengthDelta(pathfinder, 0.01)
reward = eve.reward.Combination([target_reward, path_delta])

target_reached = eve.terminal.TargetReached(intervention=intervention)
max_steps = eve.truncation.MaxSteps(200)

visualisation = eve.visualisation.SofaPygame(intervention)


env = eve.Env(
    intervention=intervention,
    start=start,
    observation=state,
    reward=reward,
    terminal=target_reached,
    truncation=max_steps,
    visualisation=visualisation,
    pathfinder=pathfinder,
)

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
