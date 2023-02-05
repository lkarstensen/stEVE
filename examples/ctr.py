import math
import numpy as np
import eve
import pygame
from eve.vesseltree import Vector
from time import perf_counter

vessel_tree = eve.vesseltree.CTRDummy()

mass_density = 1e-7
young_modulus = 1e10
device = eve.simulation3d.device.CTRTube(
    name="inner",
    outer_diameter=0.6,
    inner_diameter=0.4,
    length=300,
    tip_angle=math.pi / 2,
    tip_radius=20,
    color=[0.0, 0.0, 1.0],
    mass_density=mass_density,
    young_modulus=young_modulus,
)
device2 = eve.simulation3d.device.CTRTube(
    name="middle",
    outer_diameter=0.75,
    inner_diameter=0.61,
    length=250,
    tip_angle=math.pi / 1.5,
    tip_radius=25,
    color=[1.0, 0.0, 0.0],
    mass_density=mass_density,
    young_modulus=young_modulus,
)
device3 = eve.simulation3d.device.CTRTube(
    name="outer",
    outer_diameter=0.9,
    inner_diameter=0.76,
    length=200,
    tip_angle=math.pi / 1.5,
    tip_radius=30,
    color=[0.0, 1.0, 0.0],
    mass_density=mass_density,
    young_modulus=young_modulus,
)


simulation = eve.simulation3d.CTR(
    vessel_tree=vessel_tree, devices=[device, device2, device3], sofa_native_gui=False
)
start = eve.start.MaxDeviceLength(simulation, 200)
target = eve.target.BranchEnd(vessel_tree, simulation, threshold=10)
pathfinder = eve.pathfinder.BruteForceBFS(vessel_tree, simulation, target)
interim_target = eve.interimtarget.Even(pathfinder, simulation, target, 5, 2)
success = eve.success.InterimTargetsReached(interim_target)

insertion_length = eve.observation.InsertionLengths(simulation)
# insertion_length = eve.state.wrapper.Normalize(insertion_length)
rotation = eve.observation.Rotations(simulation)
position = eve.observation.Tracking(simulation, n_points=1)
# position = eve.state.wrapper.Normalize(position)
target_state = eve.observation.Target(interim_target)
# target_state = eve.state.wrapper.Normalize(target_state)
state = eve.observation.ObsDict([insertion_length, rotation, position, target_state])

target_reward = eve.reward.TargetReached(interim_target, factor=1.0)
step_reward = eve.reward.Step(factor=-0.002)
path_delta = eve.reward.TipToTargetDistDelta(simulation, interim_target, 0.01)
reward = eve.reward.Combination([target_reward, step_reward, path_delta])

max_steps = eve.terminal.MaxSteps(200)
target_reached = eve.terminal.TargetReached(target)
done = eve.terminal.Combination([max_steps, target_reached])

visualisation = eve.visualisation.SofaPygame(simulation, interim_target=interim_target)

randomizer = eve.vesseltreerandomizer.NewGeometry(vessel_tree=vessel_tree)

env = eve.Env(
    vessel_tree=vessel_tree,
    intervention=simulation,
    start=start,
    target=target,
    success=success,
    observation=state,
    reward=reward,
    terminal=done,
    visualisation=visualisation,
    pathfinder=pathfinder,
    vessel_tree_randomizer=randomizer,
    interim_target=interim_target,
)

r_cum = 0.0

env.reset()
last_tracking = None
while True:
    start = perf_counter()
    trans = 0.0
    rot = 0.0
    camera_trans = Vector(0.0, 0.0, 0.0)
    camera_rot = Vector(0.0, 0.0, 0.0)
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
            camera_trans += Vector(0.0, 0.0, 200.0)
        if keys_pressed[pygame.K_s]:
            camera_trans -= Vector(0.0, 0.0, 200.0)
        if keys_pressed[pygame.K_a]:
            camera_trans -= Vector(200.0, 0.0, 0.0)
        if keys_pressed[pygame.K_d]:

            camera_trans = Vector(200.0, 0.0, 0.0)
        env.visualisation.translate(camera_trans)
    if keys_pressed[pygame.K_e]:
        env.visualisation.zoom(1000)
    if keys_pressed[pygame.K_q]:
        env.visualisation.zoom(-1000)

    # trans = 10
    if keys_pressed[pygame.K_z]:
        action_1 = (trans, rot)
    elif keys_pressed[pygame.K_h]:
        action_1 = (-trans / 1.567, -rot)
    else:
        action_1 = (0, 0)
    if keys_pressed[pygame.K_u]:
        action_2 = (trans, rot)
    elif keys_pressed[pygame.K_j]:
        action_2 = (-trans / 1.567, -rot)
    else:
        action_2 = (0, 0)
    if keys_pressed[pygame.K_i]:
        action_3 = (trans, rot)
    elif keys_pressed[pygame.K_k]:
        action_3 = (-trans / 1.567, -rot)
    else:
        action_3 = (0, 0)

    action = (action_1, action_2, action_3)
    s, r, d, i, success = env.step(intervention_action=action)

    if keys_pressed[pygame.K_RETURN]:
        env.reset()
        n_steps = 0
    # print(success)
    # print(f"FPS: {1/(perf_counter()-start)}")
env.close()


# for _ in range(3):
#     print(env.reset())
#     for _ in range(10):
#         print(env.step(np.array([10, 0.5])))

# print("success")
