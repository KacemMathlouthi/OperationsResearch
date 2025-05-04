# Satellite Collision Avoidance as a Mixed-Integer Linear Programming Problem

## Problem Statement

A constellation of \$N\$ satellites must adjust their trajectories over \$T\$ time steps to:

1. Maintain safe separation \$\geq d\_{\text{safe}}\$ at all times after initial positioning
2. Minimize fuel consumption using continuous thrust control
3. Minimize deviation from nominal (no-thrust) trajectories
4. Obey orbital dynamics with realistic thrust limits

**Key Features**:

* Mixed-Integer Linear Programming (MILP) formulation
* Axis-wise collision avoidance constraints
* Joint optimization of fuel and path deviation
* Configurable temporal resolution and safety margins
* Scalable to arbitrary satellite formations

---

## Mathematical Formulation

### **Sets and Indices**

* \$\mathcal{S} = {1,...,N}\$: Set of satellites
* \$\mathcal{T} = {0,...,T}\$: Discrete time steps (nominal positions defined for \$t=0\dots T\$)
* \$\mathcal{A} = {x,y,z}\$: Coordinate axes

---

### **System Parameters**

| Symbol               | Description                                    |
| -------------------- | ---------------------------------------------- |
| \$\Delta t\$         | Time step duration                             |
| \$D\_{\text{unit}}\$ | Distance scaling factor                        |
| \$F\_{\text{max}}\$  | Maximum thrust magnitude                       |
| \$d\_{\text{safe}}\$ | Minimum safety distance                        |
| \$m\_i\$             | Satellite mass (\$\forall i \in \mathcal{S}\$) |
| \$c\_{\text{fuel}}\$ | Fuel cost coefficient                          |
| \$\lambda\$          | Deviation weight coefficient                   |

---

### **Decision Variables**

| Variable          | Description                                                         | Type       | Domain                                  |
| ----------------- | ------------------------------------------------------------------- | ---------- | --------------------------------------- |
| \$p\_{i,t}^a\$    | Position of satellite \$i\$ at time \$t\$ along axis \$a\$          | Continuous | \$\mathbb{R}\$                          |
| \$v\_{i,t}^a\$    | Velocity of satellite \$i\$ at time \$t\$ along axis \$a\$          | Continuous | \$\mathbb{R}\$                          |
| \$u\_{i,t}^a\$    | Thrust component for satellite \$i\$ at time \$t\$ along axis \$a\$ | Continuous | $\[-F\_{\text{max}}, F\_{\text{max}}]\$ |
| \$\delta\_{i,t}\$ | Fuel consumption proxy                                              | Continuous | \$\mathbb{R}^+\$                        |
| \$d\_{i,t}^a\$    | Deviation from nominal position along axis \$a\$                    | Continuous | \$\mathbb{R}^+\$                        |
| \$b\_{ij,t}^a\$   | Separation axis activation for pair \$(i,j)\$ at time \$t\$         | Binary     | \${0,1}\$                               |

---

### **Objective Function**

We seek to minimize a weighted sum of fuel usage and trajectory deviation:

```math
\text{Minimize } \sum_{i \in \mathcal{S}} \sum_{t=0}^{T-1} \delta_{i,t}
\;+
\;\lambda \sum_{i \in \mathcal{S}} \sum_{t=0}^{T} \sum_{a \in \mathcal{A}} d_{i,t}^a
```

subject to:

* \$\delta\_{i,t} \ge c\_{\text{fuel}} \sum\_{a \in \mathcal{A}} |u\_{i,t}^a|\$
* \$d\_{i,t}^a \ge |p\_{i,t}^a - p\_{i,t}^{a,\mathrm{nom}}|\$

---

### **Core Constraints**

#### **1. Orbital Dynamics**

For each satellite \$i\$, time \$t=0,...,T-1\$, and axis \$a\$:

```math
p_{i,t+1}^a = p_{i,t}^a + v_{i,t}^a \Delta t + \frac{(\Delta t)^2}{2 m_i} u_{i,t}^a \\
v_{i,t+1}^a = v_{i,t}^a + \frac{\Delta t}{m_i} u_{i,t}^a
```

#### **2. Collision Avoidance**

For all distinct satellites \$i \neq j\$ and time \$t=1,...,T\$, axis \$a\$:

```math
|p_{i,t}^a - p_{j,t}^a| \ge d_{\text{safe}} \cdot b_{ij,t}^a \\
\sum_{a \in \mathcal{A}} b_{ij,t}^a \ge 1
```

#### **3. Deviation Tracking**

For each satellite \$i\$, time \$t=0,...,T\$, and axis \$a\$:

```math
d_{i,t}^a \ge p_{i,t}^a - p_{i,t}^{a,\mathrm{nom}}, \quad
d_{i,t}^a \ge p_{i,t}^{a,\mathrm{nom}} - p_{i,t}^a
```

#### **4. Operational Limits**

```math
p_{i,0}^a = p_{i,\text{init}}^a, \quad v_{i,0}^a = v_{i,\text{init}}^a \quad \text{(Initial conditions)} \\
|u_{i,t}^a| \le F_{\text{max}} \quad \text{(Thrust bounds)}
```
