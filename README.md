# Trajectory
This repository contains routines for kinetic models, few experiments, their data generation, and training neural networks to create an efficient model for trajectory finding in plausible time frame.

### Kinetic Models and Dataset Generators
This portion of the repository is responsible for experimenting and refining kinetic models for finding trajectory data for a given borad conditions. These kinetic models are then used to generate trajectory data for different fixed initial conditions for a given board.

Run the following command to install the dependencies for the generators and kinetic models -

`!pip install z3-solver pandas matplotlib pulp gurobipy`

### AI Model Training from Dataset
This portion of the repository is responsible for training AI models on the generated trajectory data. These models are then tweaked and checked to be used later for finding trajectory data for different borad conditions.

Run the following command to install the dependencies for the neural network training models -

`!pip uninstall -y torch torchvision torchaudio`

`!pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2`

`!pip install pandas numpy scikit-learn tqdm`
