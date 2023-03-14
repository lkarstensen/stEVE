from gymnasium.envs.registration import register

import eve
import eve.visualisation

register(
    id="eve_bench/aorticarch3d",
    entry_point="eve.bench_candidates:AorticArch3D",
    kwargs={"mode": "train", "visualisation": False},
)

MAX_STEPS = 300

STEP_REWARD = -0.005
TARGET_REWARD = 1.0
PATHLENGHT_REWARD_FACTOR = 0.001

DEVICE_LENGTH_RESET = 380

DIAMETER_SCALING = 0.75
LAO_RAO_DEG = -5
CRA_CAU_DEG = 25


class AorticArch3D(eve.Env):
    def __init__(self, mode: str = "train", visualisation: bool = False) -> None:
        self.mode = mode
        self.visualisation = visualisation
        episodes_between_vessel_change = 3 if mode == "train" else 1
        vessel_tree = eve.vesseltree.AorticArchRandom(
            scale_diameter_array=[DIAMETER_SCALING],
            episodes_between_change=episodes_between_vessel_change,
        )

        device = eve.intervention.device.JWire(velocity_limit=(30, 3.14))
        simulation = eve.intervention.Intervention(
            vessel_tree=vessel_tree,
            devices=[device],
            lao_rao_deg=LAO_RAO_DEG,
            cra_cau_deg=CRA_CAU_DEG,
        )
        if mode == "train":
            start = eve.start.MaxDeviceLength(simulation, DEVICE_LENGTH_RESET)
        else:
            start = eve.start.InsertionPoint(simulation)
        target = eve.target.CenterlineRandom(
            vessel_tree,
            simulation,
            threshold=5,
            branches=[
                "rsa",
                "rcca",
                "lcca",
                "lsa",
                "bct",
                "co",
            ],
        )
        target = eve.target.filter.OutsideBranches(
            target, vessel_tree, branches_to_avoid=["aorta"]
        )

        pathfinder = eve.pathfinder.BruteForceBFS(vessel_tree, simulation, target)

        # Observation
        tracking = eve.observation.Tracking(simulation, n_points=2)
        tracking = eve.observation.wrapper.Normalize(tracking)
        tracking = eve.observation.wrapper.Memory(
            tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
        )
        target_state = eve.observation.Target(target)
        target_state = eve.observation.wrapper.ToTrackingCS(
            target_state, intervention=simulation
        )
        last_action = eve.observation.LastAction(simulation)
        last_action = eve.observation.wrapper.Normalize(last_action)
        observation = eve.observation.ObsDict(
            {
                "tracking": tracking,
                "target": target_state,
                "last_action": last_action,
            }
        )

        # Reward
        target_reward = eve.reward.TargetReached(target, factor=TARGET_REWARD)
        step_reward = eve.reward.Step(factor=STEP_REWARD)
        path_delta = eve.reward.PathLengthDelta(pathfinder, PATHLENGHT_REWARD_FACTOR)
        reward = eve.reward.Combination([target_reward, step_reward, path_delta])

        # Terminal and Truncation
        terminal = eve.terminal.TargetReached(target=target)
        truncation = eve.truncation.MaxSteps(MAX_STEPS)

        if visualisation:
            visu = eve.visualisation.SofaPygame(simulation, target=target)
        else:
            visu = None
        super().__init__(
            vessel_tree,
            simulation,
            target,
            start,
            observation,
            reward,
            terminal,
            truncation=truncation,
            pathfinder=pathfinder,
            visualisation=visu,
        )
