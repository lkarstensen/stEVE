from pynput import keyboard

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
    "w": False,
    "a": False,
    "x": False,
    "d": False,
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


keyboard_listener = keyboard.Listener(
    on_press=press_callback, on_release=release_callback
)
keyboard_listener.start()

sleep(0.01)
vessel_tree = eve.vesseltree.AorticArch(
    rotate_z=20, rotate_x=-5, omit_y_axis=True, seed=1
)
guidewire = eve.simulation2d.device.JWire(
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
catheter = eve.simulation2d.device.JWire(
    total_length=450,
    tip_length=25,
    flex_length=25,
    flex_rotary_stiffness=70000,
    flex_rotary_damping=1,
    stiff_rotary_stiffness=100000,
    stiff_rotary_damping=1,
    tip_angle=math.pi * 0.75,
    diameter=2.0,
)
physic2 = eve.simulation2d.MultiDevice(
    vessel_tree,
    [guidewire, catheter],
    velocity_limits=((50, 1.5), (50, 1.5)),
    image_frequency=7.5,
)
target = eve.target.CenterlineRandom(
    vessel_tree,
    physic2,
    5,
    branch_filter=[
        "right subclavian artery",
        "right common carotid artery",
        "left common carotid artery",
        "left subclavian artery",
        "brachiocephalic trunk",
    ],
)
pathfinder = eve.pathfinder.BruteForceBFS(vessel_tree, physic2, target)
start = eve.start.InsertionPoint(physic2)

pos = eve.observation.TrackingAllDevices(physic2)
pos = eve.observation.wrapper.Memory(
    pos, 2, reset_mode=eve.observation.wrapper.MemoryResetMode.FILL
)
pos = eve.observation.wrapper.Normalize(pos)
target_state = eve.observation.Target(target)
target_state = eve.observation.wrapper.Normalize(target_state)
action_state = eve.observation.LastAction(physic2)
action_state = eve.observation.wrapper.Memory(
    action_state, 2, reset_mode=eve.observation.wrapper.MemoryResetMode.ZERO
)
insertion_diff = eve.observation.InsertionLengthRelative(physic2, 1, 0)
insertion_diff = eve.observation.wrapper.NormalizeCustom(insertion_diff, -50, 50)
state = eve.observation.ObsDict([pos, action_state, target_state, insertion_diff])

target_reward = eve.reward.TargetReached(target, 1.0)
path_length_reward = eve.reward.PathLengthDelta(pathfinder, 0.001)
insertion_diff_reward = eve.reward.InsertionLengthRelativeDelta(
    physic2,
    1,
    0,
    -0.005,
    -30,
    10,
)
reward = eve.reward.Combination(
    [insertion_diff_reward]
)  # target_reward, path_length_reward,

done_target = eve.terminal.TargetReached(target)
done_steps = eve.terminal.MaxSteps(300)
done = eve.terminal.Combination([done_target])
success = eve.success.TargetReached(target)

randomizer = eve.vesseltreerandomizer.AorticArchRandom(
    vessel_tree, intervention=physic2, mode="train"
)

imaging = eve.imaging.ImagingDummy((500, 500))
visu = eve.visualisation.PLT2D(vessel_tree, physic2, target)

env = eve.Env(
    vessel_tree=vessel_tree,
    observation=state,
    reward=reward,
    terminal=done,
    intervention=physic2,
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
    trans_gw = 0.0
    rot_gw = 0.0
    trans_cath = 0.0
    rot_cath = 0.0
    if keys_pressed["esc"]:
        break
    if keys_pressed["up"]:
        trans_gw += 20
    if keys_pressed["down"]:
        trans_gw -= 20
    if keys_pressed["left"]:
        rot_gw += 1.5
    if keys_pressed["right"]:
        rot_gw -= 1.5
    if keys_pressed["w"]:
        trans_cath += 20
    if keys_pressed["x"]:
        trans_cath -= 20
    if keys_pressed["a"]:
        rot_cath += 1.5
    if keys_pressed["d"]:
        rot_cath -= 1.5
    # action = ((trans_gw, rot_gw), (trans_cath, rot_cath))
    action = ((trans_gw, rot_gw), (trans_cath, rot_cath))
    s, r, d, i, success = env.step(action)
    print(r)
    n_steps += 1
    r_cum += r
    env.render()
    if keys_pressed["r"]:
        env.reset()

    # print(f"fps: {1/(perf_counter()-t_loop_start)}")
    # sleep(max(physic2.dt_step - (perf_counter() - t_loop_start), 0.0))
env.close()
keyboard_listener.stop()
