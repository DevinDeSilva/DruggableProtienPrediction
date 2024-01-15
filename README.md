# DruggableProtienPrediction

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
