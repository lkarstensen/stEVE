import pickle
from typing import Any, Dict, List

import numpy as np
from ..env import Env
from ..intervention import Intervention
from ..visualisation import Visualisation, VisualisationDummy


class InterventionStateRecorder:
    def __init__(self, intervention: Intervention) -> None:
        self.intervention = intervention
        self._intervention_states = []

    def step(self):
        self._intervention_states.append(self.intervention.get_step_state())

    def reset(self):
        self._intervention_states = [self.intervention.get_reset_state()]

    def save_intervention_states(self, path: str, additional_info: Any = None):
        with open(path, "wb") as handle:
            pickle.dump(
                [self._intervention_states, additional_info],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def _dict_to_dummy(state_dict: Dict[str, Any]):
    class Dummy:
        last_action: np.ndarray

        def step(self, *args, **kwargs):
            pass

        def reset(self, *args, **kwargs):
            pass

    new_obj = Dummy()
    for key, value in state_dict.items():
        if isinstance(value, dict):
            value = _dict_to_dummy(value)
        setattr(new_obj, key, value)
    return new_obj


def _update_dummy(dummy, state_dict: Dict):
    for key, value in state_dict.items():
        if isinstance(value, dict):
            value_object = getattr(dummy, key)
            _update_dummy(value_object, value)
        else:
            value_object = value
        setattr(dummy, key, value_object)


def saved_states_to_sar(path: str, env: Env):
    with open(path, "rb") as handle:
        # [0] is states, [1] is additional info
        intervention_states: List = pickle.load(handle)[0]
    dummy_intervention = _dict_to_dummy(intervention_states.pop(0))
    env_dict = env.get_config_dict()
    new_env: Env = Env.from_config_dict(
        env_dict,
        {Intervention: dummy_intervention, Visualisation: VisualisationDummy()},
    )
    all_obs, rewards, terminals, truncations, infos, actions = [], [], [], [], [], []
    obs, info = new_env.reset()
    all_obs.append(obs)
    infos.append(info)
    for state in intervention_states:
        _update_dummy(dummy_intervention, state)
        action = dummy_intervention.last_action
        obs, reward, terminal, truncation, info = new_env.step(action)
        all_obs.append(obs)
        rewards.append(reward)
        terminals.append(terminal)
        truncations.append(truncation)
        infos.append(info)
        actions.append(action)
    return all_obs, rewards, terminals, truncations, infos, actions
