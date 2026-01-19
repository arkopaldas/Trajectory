# Introduction

This folder contains the following kinetic models. The details have been listed in the description below.

### Simplified Linear Model Trajectory Finding

Trajectory finding using simplified linear model. The velocities along each axes are independent.

*   Solver used z3.
*   Control State - (x,y) points along each axes.
*   Control Inputs - (vx, vy) velocities along each axes.
*   Dynamics -
    1.   x(t+1) = x(t) + vx(t)*dt
    2.   y(t+1) = y(t) + vy(t)*dt
    3.   L1 norm Objective function.
*   Total time taken to solve is 4.1 seconds.

### Unicycle Model Trajectory Finding

Trajectory finding using unicycle model. There is a heading orientation with a common velocity for both the axes.

*   Solver used PuLP.
*   Control State - (x,y,Θ) points along each axes and heading orientation.
*   Control Inputs - (v,Ω) velocity and angular velocity i.e. the rate of change of Θ.
*   Dynamics -
    1.   x(t+1) = x(t) + v(t)*cos(Θ(t))dt
    2.   y(t+1) = y(t) + v(t)*sin(Θ(t))dt
    3.   Θ(t+1) = Θ(t) + Ω(t)*dt
    4.   Picewise liniearization is used for continuous trigonometric dependencies.
    5.   MILP and Big-M formulation are used to reduce complexitites of constraints.
    6.   L1 norm Objective function.
*   Time taken to solve is 376.072087 seconds ~ 6 minutes 16 seconds.
