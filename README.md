# Controlling Fairness and Bias in Dynamic Learning-to-Rank
In this repository, we have uploaded the code corresponding to the SIGIR2020 Paper "Controlling Fairness and Bias in Dynamic Learning-to-Rank". 

The Paper contains 10 Figures corresponding to 10 different experiments. You can reproduce all of them from the command line. 

To start Experiment 1, call  
`python main.py with 'EXPERIMENT=1' 'PLOT_PREFIX="plots/Exp1/"'`

Experients 1-6 are based on a Real News Dataset. This can be downloaded as a csv file from [adfontes media](https://www.adfontesmedia.com/interactive-media-bias-chart/?v=402f03a963ba) and must be saved in as "InteractiveMediaBiasChart.csv" in the data subfolder.

Experients 7-10 are based on the [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/) which must be saved in as "ratings.csv" in the data subfolder.


### File Overview:

data_utils.py: For loading and preprocessing the data  
Experiments.py: Here we define the functions called for different experiments (like different starting users, different user populations, etc.)  
Documents.py: Just the classes for our Items (Movie, News article)  
relevance_network.py: The neural network for the personalized ranking  
Simulation.py : Here is the simulation happening, incl. Ranking Function, Click_Model, ...  


## Jupyter Notebook
To get started, take a look at the Jupyter Notebook "Example.ipynb". Here you can run a single trial for all our Ranking-Methods on both the News and the Movie Dataset
