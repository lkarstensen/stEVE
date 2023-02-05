from time import sleep
import eve
import random

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    vessel_tree = eve.vesseltree.AorticArch()
    device = eve.simulation3d.device.JWire()
    simulation = eve.simulation3d.MutliDeviceMP(vessel_tree, [device])

    start = eve.start.InsertionPoint(simulation)
    target = eve.target.CenterlineRandom(vessel_tree, simulation, threshold=10)
    success = eve.success.TargetReached(target)
    pathfinder = eve.pathfinder.BruteForceBFS(vessel_tree, simulation, target)

    position = eve.observation.Tracking(simulation, n_points=5)
    position = eve.observation.wrapper.RelativeToFirstRow(position)
    position = eve.observation.wrapper.CoordinatesTo2D(position, dim_to_delete="y")
    # position = eve.state.wrapper.Normalize(position)
    target_state = eve.observation.Target(target)
    target_state = eve.observation.wrapper.CoordinatesTo2D(
        target_state, dim_to_delete="y"
    )
    target_state = eve.observation.wrapper.Normalize(target_state)
    rotation = eve.observation.Rotations(simulation)
    state = eve.observation.ObsDict([position, target_state, rotation])

    target_reward = eve.reward.TargetReached(target, factor=1.0)
    step_reward = eve.reward.Step(factor=-0.01)
    path_delta = eve.reward.PathLengthDelta(pathfinder, 0.01)
    reward = eve.reward.Combination([target_reward, path_delta])

    max_steps = eve.terminal.MaxSteps(1000)
    target_reached = eve.terminal.TargetReached(target)
    done = eve.terminal.Combination([max_steps, target_reached])
    visualisation = eve.visualisation.VisualisationDummy()
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
    )

    for _ in range(3):
        env.reset()
        for _ in range(100):
            action = [[random.uniform(0, 10), random.uniform(-3.28, 3.28)]]
            s, r, d, i, success = env.step(action)
            print(s)

    env.close()
