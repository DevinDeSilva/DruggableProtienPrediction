# Druggable Protien Prediction [Report](docs/Prediction_of_Durggable_Proteins_Report.pdf)

A Druggable Protein is a protein that can interact or attach with drug-like molecules and can result in
a desired state in medicinal treatments. Therefore, identifying druggable proteins is a huge asset in the
drug industry. But just plainly identifying them using traditional experiments is costly and
time-consuming. Therefore, it is proposed to determine different features of druggable proteins that
can help identify them and then, develop machine learning models using these features to predict the
druggability of a given protein sequence

# Dataset 

## Intro
The used dataset is in text format and can be located [here](dataset) we get positive and negative samples seperately

## Preprocessing
we extract encodings from protien sequences as denoted by [Charoenkwan et al](https://www.sciencedirect.com/science/article/pii/S2589004222011555) and create several feature encodings to test.
Each encoding and the several combinations are tested to ensure the best possible feature combination. 

# Models
## Base Classifier Results.
![base_clf_res](docs/images/base_clf_results.png)

After the base classification choice an ensemble of models were made to improve the accuracy further.

## Hyperparameter Tuning.

An extensive hyperparameter tuning was conducted on the base classifiers to check their improvement.
![hyperparameters](docs/images/hyperparaeters.png)
