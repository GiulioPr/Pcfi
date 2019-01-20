# Predictor Importance by Class (PIBC) for Random Forrest Classifiers

## About PIBC
Random Forrest classifiers are accurate and effective, but often difficult to interpret. The black box of the algorithm obscures how important a feature might be to a prediction. In multiple linear regression of Linear Discriminant Analysis, we can see the impact of a predictor on a prediction by the magnitude of the predictor's $\beta$. In the Random Forrest, we can evaluate a variable's importance _to the model_ by its Mean Decrease in Gini Index, but we cannot see how important a given variable is to correct classification of a single _class_. The PIBC provides a method to calculate the importance of each predictor in determining which class an observation belongs to. PIBC treats the regression tree as a graph, and calculating the purity of each split produced by a single predictor in all walks from the terminal leaf nodes of a class to the root.

## Data used in project
[Wine dataset](https://archive.ics.uci.edu/ml/datasets/wine) taken from the UCI Machine Learning Repository.