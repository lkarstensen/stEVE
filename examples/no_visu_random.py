import random
import logging
import eve

if __name__ == "__main__":
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        eve.intervention.vesseltree.ArchType.VII
    )
    device = eve.intervention.device.JShaped()

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
        devices=[device],
        simulation=simulation,
        fluoroscopy=fluoroscopy,
        target=target,
    )

    start = eve.start.MaxDeviceLength(intervention=intervention, max_length=500)
    pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)

    position = eve.observation.Tracking2D(intervention=intervention, n_points=5)
    position = eve.observation.wrapper.NormalizeTracking2DEpisode(
        position, intervention
    )
    target_state = eve.observation.Target2D(intervention=intervention)
    target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
        target_state, intervention
    )
    rotation = eve.observation.Rotations(intervention=intervention)

    state = eve.observation.ObsDict(
        {"position": position, "target": target_state, "rotation": rotation}
    )

    target_reward = eve.reward.TargetReached(
        intervention=intervention,
        factor=1.0,
    )
    path_delta = eve.reward.PathLengthDelta(
        pathfinder=pathfinder,
        factor=0.01,
    )
    reward = eve.reward.Combination([target_reward, path_delta])

    target_reached = eve.terminal.TargetReached(intervention=intervention)
    max_steps = eve.truncation.MaxSteps(200)
    env = eve.Env(
        intervention=intervention,
        start=start,
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
