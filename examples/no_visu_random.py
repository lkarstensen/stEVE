import random
import logging
import eve

if __name__ == "__main__":
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    vessel_tree = eve.vesseltree.AorticArch(eve.vesseltree.ArchType.VII)
    device = eve.intervention.device.JShaped()
    simulation = eve.intervention.MonoPlaneStatic(
        vessel_tree, [device], sofacore_mp=True, mp_timeout_step=1, image_frequency=7.5
    )

    start = eve.start.InsertionPoint(simulation)
    target = eve.target.CenterlineRandom(vessel_tree, simulation, threshold=10)
    pathfinder = eve.pathfinder.BruteForceBFS(vessel_tree, simulation, target)

    position = eve.observation.Tracking2D(simulation, n_points=5)
    position = eve.observation.wrapper.RelativeToFirstRow(position)
    target_state = eve.observation.Target2D(target)
    target_state = eve.observation.wrapper.Normalize(target_state)
    rotation = eve.observation.Rotations(simulation)
    state = eve.observation.ObsDict(
        {"position": position, "target": target_state, "rotation": rotation}
    )

    target_reward = eve.reward.TargetReached(target, factor=1.0)
    step_reward = eve.reward.Step(factor=-0.01)
    path_delta = eve.reward.PathLengthDelta(pathfinder, 0.01)
    reward = eve.reward.Combination([target_reward, path_delta])

    target_reached = eve.terminal.TargetReached(target=target)
    max_steps = eve.truncation.MaxSteps(200)
    env = eve.Env(
        vessel_tree=vessel_tree,
        intervention=simulation,
        start=start,
        target=target,
        observation=state,
        reward=reward,
        terminal=target_reached,
        truncation=max_steps,
        pathfinder=pathfinder,
    )

    for _ in range(3):
        env.reset()
        print("reset")
        for _ in range(10):
            action = [[random.uniform(0, 10), random.uniform(-3.28, 3.28)]]
            s, r, term, trunc, i = env.step(action)
            print(s)

    env.close()
