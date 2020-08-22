# Per Class Feature Importance (PCFI) for Classifiers based on Decision Tree Models

Machine learning models are accurate and effective, but often difficult to interpret.
Here, we define a method to explain decision tree-based classifiers named Per Class Feature Importance (PCFI), offering an alternative to SHAP values.
As the name suggests, PCFI builds upon the default feature importance, which is calculated from the mean impurity decrease of the feature's splits in the decision tree.

Our aim is to provide a quantitative description for the contribution of each feature to the prediction of one specific class label, thus returning a more detailed description on the model structure and handling of the input data.


### PCFI Definition

The underlying idea is simple.
The importance of a given feature to a certain class is the sum of the mean impurity decreases from those (and only those) nodes splitting on that feature and leading to a leaf node that predicts such class (thinking the model as a directed tree graph only with paths from the initial root node to the leaves).
In particular, the mean impurity decreases in one node is calculated as if the variables in had a binary label: 1 if the variable belongs to the class under consideration, 0 otherwise.
In the case of ensemble methods, such as Random Forest Classifiers, the PCFI is obtained averaging the importance, for one combination of class and feature, across all trees in the ensemble.

In this way, PCFI coincides with the default feature importance in the case of binary classification.
Moreover, each vector of feature importances, associated to a given class, effectively provides a measure of how well the model can discriminate such class from the others.  

A measure of importance that considers each class separately is particularly relevant to imbalanced datasets, as features that are specific to rare classes would not emerge with the default feature importance. Of note, the latter only provides a ?global? view of the model.

In fact, PCFI was inspired during the analysis of biology data, where rare classes (e.g. different cell types) often are the main subject of investigation.
This is often the case in cell type discovery studies, or during the analyses of rare cell types as hematopoietic stem cells in the murine bone marrow.


### Code

The demo for PCFI is found in 1_code/Dermatology_dataset.ipynb, where we used a dermatology dataset from the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/) to show the improvement of PCFI with respect to the default importance method.
Thus, following a biology-oriented approach, we demonstrate how PCFI can be used to rank and select for features that are relevant to targeted classes. In biology, this is useful to inform how to restrict a panel of markers (i.e. features), while maintaining good discriminating power with respect to selected targets (i.e. classes).

Finally, in 1_code/Benchmark_feature_ranking.ipynb, we compare three methods for feature ranking: global (or default) feature importance, SHAP values and PCFI; the last two are also compared when ranking features that are relevant to one specific rare class.
Results from this comparison are not conclusive, as SHAP values and PCFI achieve similar results on the 9 dataset we considered.
