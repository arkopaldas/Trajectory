# Unicycle Model Trajectory Finding Experiment

Trajectory finding using unicycle model. There is a heading orientation with a common velocity for both the axes.

*   Solver used z3.
*   Control State - (x,y,Θ) points along each axes and heading orientation.
*   Control Inputs - (v,Ω) velocity and angular velocity i.e. the rate of change of Θ.
*   Dynamics -
    1.   x(t+1) = x(t) + v(t)*cos(Θ(t))dt
    2.   y(t+1) = y(t) + v(t)*sin(Θ(t))dt
    3.   Θ(t+1) = Θ(t) + Ω(t)*dt
    4.   Picewise liniearization is used for continuous trigonometric dependencies.
    5.   L1 norm Objective function.
*   Only a single obstacle is placed otherwise no solutions were acheived.
*   Total time taken to solve is 13.5 seconds.

# Unicycle Model Trajectory Finding with Fixed Initial Condition Experiment

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
*   The initial state is fixed arbitrarily.
*   Time taken to solve is 6652.260 seconds ~ 01 hour 50 minutes 52 seconds.
