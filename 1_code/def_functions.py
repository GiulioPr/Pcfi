import numpy as np
import pandas as pd
import seaborn as sns
import copy as cp
import shap
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris, load_wine
from matplotlib import pyplot as plt



def f_gini(v):
    p = v/v.sum()
    return np.multiply(p, 1-p).sum()



def f_entropy(v):
    return sps.entropy(pk=v)



def f_misclassification(v):
    p = v/v.sum()
    return 1 - p.max()



def calcImportanceMatrix(rf_clf):
    """
    Calculates the importance matrix of predictors for each class.

    Arguments:
        rf_clf - The random forest classifier to calculate the importance matrix for.
        Must (?) be a RandomForestClassifier object. EXTENSIBLE TO ANY DECISION TREE CLASSIFIER FROM sklearn.tree?

    Returns:
        importance_matrix - The importance matrix with the importance of each predictor in predicting a class.
        A n by m numpy array.

    """

    # get the number of classes being predicted by the random forest
    classes = rf_clf.classes_
    n_classes = len(classes)

    # init storage for the predictor importances by classes by trees
    importance_matrix = []

    #dec_tree = rf_clf.estimators_[0]
    for dec_tree in rf_clf.estimators_:

        # get the criterion used to measure impurity
        criterion = dec_tree.get_params()['criterion']
        if criterion == 'gini':
            f_impurity = f_gini
        elif criterion == 'entropy':
            f_impurity = f_entropy
        elif criterion == 'misclassification':
            f_impurity = f_misclassification
        else:
            f_impurity = 0
            print('Unassigned impurity measure')

        # get the number of features and nodes in the tree
        feature = dec_tree.tree_.feature
        n_features = dec_tree.tree_.n_features
        n_nodes = dec_tree.tree_.__getstate__()['node_count']
        nodes = dec_tree.tree_.__getstate__()['nodes']
        parent_node_ind = -np.ones(shape=n_nodes, dtype='<i8')
        #parent_node_ind[0] = n_nodes + 1
        #print(parent_node_ind)
        for par_ind,node in enumerate(nodes):
            if node[0] != -1:
                parent_node_ind[node[0]] = par_ind
            if node[1] != -1:
                parent_node_ind[node[1]] = par_ind
        #print(parent_node_ind)

        # identify the leaves of the tree
        is_leaves = np.array([node[0]==-1 and node[1]==-1 for node in nodes])
        leaves_index = np.nonzero(is_leaves)[0]

        values_sorted = dec_tree.tree_.__getstate__()['values']
        #print ('nodes', nodes, len(nodes), len(values_sorted[:,0,:]))
        node_pred = np.argmax(values_sorted[:,0,:], axis=1)
        leaves_class_index = node_pred[is_leaves]
        #for par_ind,node in enumerate(nodes):
        #    print(par_ind,parent_node_ind[par_ind],is_leaves[par_ind],node,
        #          values_sorted[par_ind], values_sorted[par_ind].sum())
        node_unvisited = np.ones((n_classes, n_nodes), dtype=bool)
        tree_importances = np.zeros((n_classes, n_features))
        for leaf_i,leaf_c_i in zip(leaves_index,leaves_class_index):
            current_i = parent_node_ind[leaf_i]
            #print('START from leaf ', leaf_i, 'with class ', leaf_c_i)
            #print('whose parent is ', current_i)
            # walk the tree and calculate the importance of the predictor
            while current_i != -1 and node_unvisited[leaf_c_i,current_i]:
                current_node = nodes[current_i]
                left_node = nodes[current_node['left_child']]
                right_node = nodes[current_node['right_child']]
                current_feature = current_node['feature']
                ###NEW HERE
                current_values = values_sorted[current_i,0,:]
                left_values = values_sorted[current_node['left_child'],0,:]
                right_values = values_sorted[current_node['right_child'],0,:]

                current_values_class = np.array([
                    current_values[leaf_c_i],
                    current_values[np.arange(len(current_values)) != leaf_c_i].sum()
                ])
                left_values_class = np.array([
                    left_values[leaf_c_i],
                    left_values[np.arange(len(left_values)) != leaf_c_i].sum()
                ])
                right_values_class = np.array([
                    right_values[leaf_c_i],
                    right_values[np.arange(len(right_values)) != leaf_c_i].sum()
                ])
                #print(
                #    current_values,
                #    np.array([current_values[leaf_c_i],
                #              current_values[np.arange(len(current_values))!=leaf_c_i].sum()])
                #     )
                #print(current_values.sum(), left_values.sum(), right_values.sum(),
                #     left_values.sum()/current_values.sum(), right_values.sum()/current_values.sum(),
                #      current_node['weighted_n_node_samples'], left_node['weighted_n_node_samples'],
                #      right_node['weighted_n_node_samples']
                #     )
                tree_importances[leaf_c_i,current_feature] += (
                        current_node['weighted_n_node_samples'] * f_impurity(current_values_class) -
                        left_node['weighted_n_node_samples'] * f_impurity(left_values_class) -
                        right_node['weighted_n_node_samples'] * f_impurity(right_values_class)
                        )
                #print('\n', current_node, (
                #        current_node['weighted_n_node_samples'] * f_importance(current_values) -
                #        left_node['weighted_n_node_samples'] * f_importance(left_values) -
                #        right_node['weighted_n_node_samples'] * f_importance(right_values)
                #        ))
                ###
                node_unvisited[leaf_c_i,current_i] = False
                current_i = parent_node_ind[current_i]
                #print('next current is ', current_i)
        importance_matrix.append(tree_importances/nodes[0]['weighted_n_node_samples'])

    # average the predictor importances for each class by all of the trees in the forest
    importance_matrix = np.mean(importance_matrix, axis = 0)
    #normalise importance over each class
    importance_matrix = (importance_matrix.T / np.sum(importance_matrix, axis=1)).T
    return(importance_matrix)



def plot_confusion_matrix(y_true, y_pred, classes,
                          title='Normalized confusion matrix',
                          cmap=plt.cm.plasma):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, label='Percentage')
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.0f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(round(cm[i, j],0), fmt),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "white")
    plt.tight_layout()
    return fig, ax



def top_features(rf_clf, feature_names, X_train, y_train, y):
    #Global importance ranking
    global_importance = rf_clf.feature_importances_
    global_index = np.argsort(global_importance)
    global_index = global_index[::-1]
    
    #Shap values ranking
    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X_train)
    shap_imp_mat = np.array([np.mean(abs(class_shap), axis=0) for class_shap in shap_values])
    ###NORMALIZE ROWS
    row_sums = shap_imp_mat.sum(axis=1)
    shap_imp_mat = shap_imp_mat / row_sums[:, np.newaxis]
    ###
    shap_imp = np.mean(shap_imp_mat, axis=0)
    shap_index = np.argsort(shap_imp)
    shap_index = shap_index[::-1]

    #Per-class importance ranking
    imp_mat = calcImportanceMatrix(rf_clf)
    mean_importance = np.mean(imp_mat, axis=0)
    pcfi_index = np.argsort(mean_importance)
    pcfi_index = pcfi_index[::-1]

    ###check y_train rare class
    unique_classes, count_classes = np.unique(y_train, return_counts=True)
    rare_class_index = np.argsort(count_classes)[0]
    rare_class_train = unique_classes[rare_class_index]
    
    unique_classes, count_classes = np.unique(y, return_counts=True)
    rare_class_index = np.argsort(count_classes)[0]
    rare_class = unique_classes[rare_class_index]
    print('Do y and y_train have same rare class?', rare_class==rare_class_train)
    
    rare_class_pcfi_index = np.argsort(imp_mat[rare_class_index])
    rare_class_pcfi_index = rare_class_pcfi_index[::-1]
    
    rare_class_shap_index = np.argsort(shap_imp_mat[rare_class_index])
    rare_class_shap_index = rare_class_shap_index[::-1]
    
    ##STOP AT THE FIRST DIFFERING INSTANCE
    index = [i for i in np.arange(len(feature_names)) if shap_index[i] != pcfi_index[i]]#SHAP OR GLOBAL?
    if len(index) > 0 and index[0] > 2:
        top_global_features = feature_names[global_index[:index[0]+1]]
        top_shap_features = feature_names[shap_index[:index[0]+1]]
        top_pcfi_features = feature_names[pcfi_index[:index[0]+1]]
    else:
        top_global_features = feature_names[global_index[:3]]
        top_shap_features = feature_names[shap_index[:3]]
        top_pcfi_features = feature_names[pcfi_index[:3]]
    top_rare_pcfi_features = feature_names[rare_class_pcfi_index[:3]]
    top_rare_shap_features = feature_names[rare_class_shap_index[:3]]
    
    if all(lab in top_rare_pcfi_features for lab in top_rare_shap_features):
        top_rare_shap_features = top_rare_pcfi_features
    #REORDER EQUAL LABELS
    top_features_lst = [top_global_features, top_shap_features, top_pcfi_features, 
                        top_rare_shap_features, top_rare_pcfi_features]
    for k in np.arange(2):
        f1 = top_features_lst[k]
        for j in np.arange(k+1,3):
            f2 = top_features_lst[j]
            if all(lab in f1 for lab in f2):
                top_features_lst[j] = f1
    return top_features_lst, rare_class



def fit_acc_score(y_test, predictions):
    acc = 100 * accuracy_score(y_test, predictions)
    return round(acc, 2)#, round(f1score, 2))#round(np.mean(errors), 2),



def fitRF_and_rank(data, feature_names, class_col, rnd_seed=45):#class_names, class_tags, 
    X = np.array(data.loc[:,feature_names])
    y = np.array(data.loc[:,class_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rnd_seed)
    
    #Print dataset info and check that all classes are represented in  y_train and y_test 
    unique_classes, count_classes = np.unique(y, return_counts=True)
    is_balanced = all([i==j for i in count_classes for j in count_classes])
    print('Is the dataset balanced?', is_balanced)
    print('Classes and counts:', unique_classes, count_classes)
    print('Classes count in train:', np.unique(y_train, return_counts=True))
    print('Classes count in test:', np.unique(y_test, return_counts=True))
    
    # Train the classifier
    rf_gscv = GridSearchCV(estimator=RandomForestClassifier(random_state=rnd_seed),
                          param_grid={'n_estimators':[10, 25, 50, 75, 100, 150, 200, 250, 300],
                                      'max_depth': [10, 15, 20, 25, None]},
                          scoring='accuracy', cv=3, iid=False)
    rf_gscv.fit(X_train, y_train)
    print('Best score:', rf_gscv.best_score_)
    rf_clf = rf_gscv.best_estimator_
          
    predictions = rf_clf.predict(X_test)
    accuracy = fit_acc_score(y_test, predictions)
    print('Model performances:')
    print('Accuracy: {}'.format(accuracy))
    top_feature_lst, rare_class = top_features(rf_clf, feature_names, X_train, y_train, y)
    return (rf_gscv, top_feature_lst, rare_class)



def refitRF(data, feature_names, class_col, rf_gscv, rare_class, rnd_seed=45, global_score=False):
    X = np.array(data.loc[:,feature_names])
    y = np.array(data.loc[:,class_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rnd_seed)
    
    # Train the classifier
    rf_clf = RandomForestClassifier(random_state=rnd_seed,
                                    max_depth=rf_gscv.best_params_['max_depth'],
                                    n_estimators=rf_gscv.best_params_['n_estimators']
                                   )
    rf_clf.fit(X_train, y_train)
          
    predictions = rf_clf.predict(X_test)
    if global_score:
        score = fit_acc_score(y_test, predictions)
    else:
        score = round(100 * f1_score(y_test, predictions, labels=[rare_class], average=None)[0], 2)
    return score