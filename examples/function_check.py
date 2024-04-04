# pylint: disable=no-member

from time import perf_counter
import numpy as np
import eve
from eve.visualisation.sofapygame import SofaPygame


def delete_lines(n=1):
    for _ in range(n):
        print("\033[1A\x1b[2K", end="")


# Define Intervention
vessel_tree = eve.intervention.vesseltree.AorticArch(
    seed=30,
    scaling_xyzd=[1.0, 1.0, 1.0, 0.75],
)


device = eve.intervention.device.JShaped()

simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)

fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
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
    devices=[device],
    simulation=simulation,
    fluoroscopy=fluoroscopy,
    target=target,
)

# Helper Objects
start = eve.start.MaxDeviceLength(intervention=intervention, max_length=500)
pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)

# Define Observation
position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
position = eve.observation.wrapper.NormalizeTracking2DEpisode(position, intervention)
target_state = eve.observation.Target2D(intervention=intervention)
target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
    target_state, intervention
)
rotation = eve.observation.Rotations(intervention=intervention)
state = eve.observation.ObsDict(
    {"position": position, "target": target_state, "rotation": rotation}
)

# Define Reward
target_reward = eve.reward.TargetReached(
    intervention=intervention,
    factor=1.0,
)
path_delta = eve.reward.PathLengthDelta(
    pathfinder=pathfinder,
    factor=0.01,
)
reward = eve.reward.Combination([target_reward, path_delta])


# Define Terminal and Truncation
target_reached = eve.terminal.TargetReached(intervention=intervention)
max_steps = eve.truncation.MaxSteps(200)

# Add Visualisation
visualisation = SofaPygame(intervention=intervention)

# Combine everything in an env
env = eve.Env(
    intervention=intervention,
    observation=state,
    reward=reward,
    terminal=target_reached,
    truncation=max_steps,
    visualisation=visualisation,
    start=start,
    pathfinder=pathfinder,
)


n_steps = 0
r_cum = 0.0

env.reset()

for _ in range(12):
    print("")

for _ in range(50):
    start = perf_counter()
    trans = 35.0
    rot = 1.0
    camera_trans = np.array([0.0, 0.0, 0.0])
    lao_rao = 10
    action = (trans, rot)
    env.visualisation.rotate(lao_rao, 0)
    obs, reward, terminal, truncation, info = env.step(action=action)
    env.render()
    n_steps += 1

    delete_lines(11)
    print(f"Observation: \n {obs}\n")
    print(f"Reward: {reward:.2f}\n")
    print(f"FPS: {1/(perf_counter()-start):.2f} Hz")
env.close()
