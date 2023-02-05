from pynput import keyboard

import eve

import eve
from time import sleep, time, perf_counter
import math

keys_pressed = {
    "up": False,
    "down": False,
    "right": False,
    "left": False,
    "esc": False,
    "r": False,
    "c": False,
}


def press_callback(key):
    if isinstance(key, keyboard.Key):
        key = key.name
    else:
        key = key.char
    if key in keys_pressed.keys():
        keys_pressed[key] = True


def release_callback(key):
    if isinstance(key, keyboard.Key):
        key = key.name
    else:
        key = key.char
    if key in keys_pressed.keys():
        keys_pressed[key] = False


l = keyboard.Listener(on_press=press_callback, on_release=release_callback)
l.start()

sleep(0.01)
tree = eve.vesseltree.AorticArch(rotate_z=20, rotate_x=-5, omit_y_axis=True)


instrument = eve.simulation2d.device.JWire(
    total_length=450,
    tip_length=15.2,
    flex_length=15.2,
    flex_rotary_stiffness=70000,
    flex_rotary_damping=1,
    stiff_rotary_stiffness=100000,
    stiff_rotary_damping=1,
    tip_angle=math.pi * 0.4,
    diameter=0.9,
)
simu = eve.simulation2d.SingleDevice(
    vessel_tree=tree,
    device=instrument,
    element_length=3.5,
    friction=1.0,
    velocity_limit=(50, 3.14),
    image_frequency=7.5,
    dt_simulation=2.5 / 10000,
    stop_device_at_tree_end=True,
)
target = eve.target.CenterlineRandom(
    tree,
    simu,
    5,
    branch_filter=[
        "right subclavian artery",
        "right common carotid artery",
        "left common carotid artery",
        "left subclavian artery",
        "brachiocephalic trunk",
    ],
)
pathfinder = eve.pathfinder.BruteForceBFS(tree, simu, target)
start = eve.start.MaxDeviceLength(simu, 380)
imaging = eve.imaging.LNK1(simu, (500, 500), vessel_tree=tree)

pos = eve.observation.Tracking(simu)
pos = eve.observation.wrapper.Memory(
    pos, 3, reset_mode=eve.observation.wrapper.MemoryResetMode.FILL
)
pos = eve.observation.wrapper.Normalize(pos)
target_state = eve.observation.Target(target)
target_state = eve.observation.wrapper.Normalize(target_state)
action_state = eve.observation.LastAction(simu)
action_state = eve.observation.wrapper.Memory(
    action_state, 2, reset_mode=eve.observation.wrapper.MemoryResetMode.ZERO
)
image = eve.observation.Image(imaging)
state = eve.observation.ObsDict([pos, action_state, target_state])

target_reward = eve.reward.TargetReached(target, 1.0)
path_length_reward = eve.reward.PathLengthDelta(pathfinder, 0.001)
reward = eve.reward.Combination([target_reward, path_length_reward])
done_target = eve.terminal.TargetReached(target)
done_steps = eve.terminal.MaxSteps(300)
done = eve.terminal.Combination([done_target])
visu = eve.visualisation.FromImaging(imaging)
success = eve.success.TargetReached(target)

randomizer = eve.vesseltreerandomizer.AorticArchRandom(
    tree, intervention=simu, mode="train", diameter_scaling_range=[0.6, 0.6]
)
env = eve.Env(
    vessel_tree=tree,
    observation=image,
    reward=reward,
    terminal=done,
    intervention=simu,
    start=start,
    target=target,
    imaging=imaging,
    pathfinder=pathfinder,
    visualisation=visu,
    success=success,
    vessel_tree_randomizer=randomizer,
)

env.reset()


n_steps = 0
r_cum = 0.0
while True:
    t_loop_start = perf_counter()
    trans = 0.0
    rot = 0.0
    contrast = 0.0
    if keys_pressed["esc"]:
        break
    if keys_pressed["up"]:
        trans += 20
    if keys_pressed["down"]:
        trans -= 20
    if keys_pressed["left"]:
        rot += 15
    if keys_pressed["right"]:
        rot -= 15
    action = (trans, rot)
    if keys_pressed["c"]:
        contrast = 1.0

    s, r, d, i, success = env.step(action, contrast)
    n_steps += 1
    r_cum += r
    env.render()
    if keys_pressed["r"]:
        env.reset()

    print(f"fps: {1/(perf_counter()-t_loop_start)}")
    # sleep(max(physic2.dt_step - (perf_counter() - t_loop_start), 0.0))
env.close()
l.stop()
