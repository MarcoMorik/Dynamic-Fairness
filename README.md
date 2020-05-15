# Dynamic-Fairness
Dynamic Fair Rankings



To start Experiment 1, call  
`python main.py with 'EXPERIMENT=1' 'PLOT_PREFIX="plots/Exp1/"'`

There are 10 Experiments implemented in the main function, corresponding to the 10 Figures of the Paper.


### File Overview:


data_utils.py: For loading and preprocessing the data
Experiments.py: Here we define the functions called for different experiments (like different starting users, different user populations, etc.)
Documents.py: Just the classes for our Items (Movie, News article)
relevance_network.py: The neural network for the personalized ranking 
Simulation.py : Here is the simulation happening, incl. Ranking Function, Click_Model, ...
