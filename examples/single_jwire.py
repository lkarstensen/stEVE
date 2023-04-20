# pylint: disable=no-member

import os
from time import perf_counter
import pygame
import numpy as np
import eve
from eve.visualisation.sofapygame import SofaPygame
import eve.bench_candidates


vessel_tree = eve.vesseltree.AorticArch(
    seed=30,
    scale_xyzd=[1.0, 1.0, 1.0, 0.75],
    rotate_yzx_deg=[0, -20, -5],
)


device = eve.intervention.device.JWire()

simulation = eve.intervention.Simulation(
    vessel_tree=vessel_tree,
    devices=[device],
    stop_device_at_tree_end=True,
)
start = eve.start.MaxDeviceLength(
    intervention=simulation,
    max_length=500,
)
target = eve.target.CenterlineRandom(
    vessel_tree=vessel_tree,
    intervention=simulation,
    threshold=5,
    branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
)
pathfinder = eve.pathfinder.BruteForceBFS(
    vessel_tree=vessel_tree,
    intervention=simulation,
    target=target,
)

position = eve.observation.Tracking2D(
    intervention=simulation,
    n_points=5,
)
position = eve.observation.wrapper.NormalizeTracking2DEpisode(position, simulation)
target_state = eve.observation.Target2D(target=target)
target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
    target_state, simulation
)
rotation = eve.observation.Rotations(intervention=simulation)

state = eve.observation.ObsDict(
    {"position": position, "target": target_state, "rotation": rotation}
)

target_reward = eve.reward.TargetReached(
    target=target,
    factor=1.0,
)
path_delta = eve.reward.PathLengthDelta(
    pathfinder=pathfinder,
    factor=0.01,
)
reward = eve.reward.Combination([target_reward, path_delta])


target_reached = eve.terminal.TargetReached(target=target)
max_steps = eve.truncation.MaxSteps(200)

imaging = eve.imaging.Pillow(simulation, (1000, 2000))

visualisation = SofaPygame(intervention=simulation, target=target)


env = eve.Env(
    vessel_tree=vessel_tree,
    intervention=simulation,
    start=start,
    target=target,
    observation=state,
    reward=reward,
    terminal=target_reached,
    truncation=max_steps,
    visualisation=visualisation,
    pathfinder=pathfinder,
    imaging=imaging,
)

# folder = os.path.dirname(os.path.abspath(__file__))
# config_path = os.path.join(folder, "env_config.yml")
# env.save_config(config_path)


# env = eve.Env.from_config_file(config_path)


n_steps = 0
r_cum = 0.0

env.reset()

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
        trans += 50
    if keys_pressed[pygame.K_DOWN]:
        trans -= 50
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
    action = (trans, rot)
    obs, reward, terminal, truncation, info = env.step(action=action)
    env.render()
    n_steps += 1
    print(obs)
    if keys_pressed[pygame.K_RETURN]:
        env.reset()
        n_steps = 0

    print(f"FPS: {1/(perf_counter()-start)}")
env.close()
