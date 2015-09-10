#The Cell Line Classification Project
This is a project that attempts to classify cell lines as either sensitive or resistant to a new class of drugs known as SMAPs.
SMAPs is an acronym that stands for "Small Molecular Activators of PP2A."

We have some inital IC50 data that tells us how certain cell lines have reacted to SMAPs.
We are using this data, along with publicly available gene expression data from CCLE to train machine learning models
to classify new cell lines as either sensitive or resistant to SMAPs.

##What exactly are we trying to do in this project?
1. Train Support Vector Machine (SVM) models on our training data and use them to predict both cell line and patient reaction to SMAPs.
2. Train Feed-Forward Neural Network (FNN) models on our training data and use them to predict both cell line and patient reaction to SMAPs.
3. Train the NEAT algorithm (Neuro Evolution of Augmented Topologies) and use it to predict both cell line and patient reaction to SMAPs.

##How to run the code
Install Python 2.7 with the following libraries: scikit-learn, pybrain, pandas, scipy, numpy, and argparse
Clone the git repository
`git clone https://github.com/joewledger/Cell-Line-Classification.git`
From the main directory run the following command to run the program with default parameters:
`python Src/Result_Writer.py`
To see what parameters are avaliable, type the following:
`python Src/Result_Writer.py --help`
Enjoy!!


