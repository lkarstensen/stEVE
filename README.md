# EVE (Endovascular Environment)
Framework for the creation of simulations of endovascular interventions using the [SOFA simulation framework](https://www.sofa-framework.org) as simulation engine. EVE was designed for reinforcement learning and offers the seamless integration of *state*, *reward*, *terminal*, *truncation* and *info* features as defined by the [Farama gymnasium](https://gymnasium.farama.org) and can therefore be integrated with any RL framework adhering to this standard. Nevertheless, it is possible to use the simulated intervention for other purposes.

During design high priorities were modularity and pythonic way of usage. Resulting in the following architecture. 

[<img src="https://github.com/lkarstensen/eve/blob/main/figures/eve_architecture.png" width="500"/>](https://github.com/lkarstensen/eve/blob/main/figures/eve_architecture.png)

## Getting started

Endovascular Environment: Modular Toolbox to quickly prototype new Endovascular Environments

This Repo allows to create modular endovascular environments with the following components:

* Vesseltree
* Device
* Simulation
* Start position of episode (within bounds)
* Target selection
* Pathfinder
* Observation
* Reward
* Terminal criterion
* Truncation criterion
* Imaging (Optional)
* Interim Targets (Optional)
* Visualization (Optional)
