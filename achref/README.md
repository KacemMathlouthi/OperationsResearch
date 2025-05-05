# Satellite Collision Avoidance Web App

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A web-based interface for optimizing satellite trajectories to prevent collisions while minimizing fuel consumption and deviation from uncorrected paths. Built with Python/Gurobi for optimization and Three.js for 3D visualization.

![App Screenshot](./assets/satellite_trajectories.gif)

### Optimization Objective

We now jointly minimize fuel usage and trajectory deviation from the nominal (no-thrust) path:

```math
\text{Minimize } \sum_{i=1}^N \sum_{t=0}^{T-1} \delta_{i,t}
\;+
\;\lambda \sum_{i=1}^N \sum_{t=0}^{T} \sum_{a\in\{x,y,z\}} d_{i,t}^a
```

where:

* \$\delta\_{i,t} \ge c\_{\text{fuel}}\sum\_{a\in{x,y,z}} |u\_{i,t}^a|\$ (fuel proxy)
* \$d\_{i,t}^a \ge |p\_{i,t}^a - p\_{i,t}^{a,\mathrm{nom}}|\$ (deviation from nominal trajectory)
* \$\lambda\$ is an optional weight balancing fuel vs. deviation (default \$\lambda=1\$).

### Features

* Interactive 3D visualization of satellite trajectories
* Configurable safety margins, thrust limits, and deviation weight
* Real-time MILP optimization backend
* Scenario export/import in JSON format

## Problem Formulation

### Orbital Dynamics

For satellite \$i\$ at time \$t\$ with mass \$m\_i\$ and thrust \$\mathbf{u}\_{i,t}\$:

```math
\begin{aligned}
\mathbf{p}_{i,t+1} &= \mathbf{p}_{i,t} + \mathbf{v}_{i,t} \Delta t + \frac{1}{2m_i} \mathbf{u}_{i,t} (\Delta t)^2 \\
\mathbf{v}_{i,t+1} &= \mathbf{v}_{i,t} + \frac{1}{m_i} \mathbf{u}_{i,t} \Delta t
\end{aligned}
```

### Collision Avoidance

For all satellite pairs \$(i,j)\$ and times \$t \ge 1\$:

```math
\begin{aligned}
|p_{i,t}^x - p_{j,t}^x| &\ge d_{\text{safe}} \cdot b_{ij,t}^x \\
|p_{i,t}^y - p_{j,t}^y| &\ge d_{\text{safe}} \cdot b_{ij,t}^y \\
|p_{i,t}^z - p_{j,t}^z| &\ge d_{\text{safe}} \cdot b_{ij,t}^z \\
 b_{ij,t}^x + b_{ij,t}^y + b_{ij,t}^z &\ge 1 \quad (b_{ij,t}^a \in \{0,1\})
\end{aligned}
```

### Additional Resources

For a comprehensive explanation of the problem statement, including mathematical formulations and assumptions, refer to the [Problem Statement](./assets/problem_statement.md). This document provides in-depth details about the orbital dynamics, collision avoidance constraints, and optimization objectives used in this project.
