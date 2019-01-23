# function to generate the importance matrix (e.g. how important is each predictor to predicting each class?)

import numpy as np

def calcImportanceMatrix(rf_clf):
    """
    Calculates the importance matrix of predictors for each class.

    Arguments:
        rf_clf - The random forest classifier to calculate the importance matrix for. Must be a RandomForestClassifier object.

    Returns:
        importance_matrix - The importance matrix with the importance of each predictor in predicting a class. A n by m numpy array.

    """

    # get the number of classes being predicted by the random forest
    classes = rf_clf.classes_
    n_classes = len(classes)

    # init storage for the predictor importances by classes by trees
    importance_matrix = []

    for dec_tree in rf_clf.estimators_:
        
        # get the number of features used in the tree
        feature = dec_tree.tree_.feature
        n_features = dec_tree.tree_.n_features
        n_nodes = dec_tree.tree_.__getstate__()['node_count']
        nodes = dec_tree.tree_.__getstate__()['nodes']
        parent_node_ind = -np.ones(shape=n_nodes, dtype='<i8')
        
        for par_ind,node in enumerate(nodes):
            parent_node_ind[node[0]] = par_ind
            parent_node_ind[node[1]] = par_ind

        # identify the leaves of the tree
        is_leaves = np.array([node[0]==node[1] for node in nodes])
        leaves_index = np.nonzero(is_leaves)[0]

        leaves_parent = parent_node_ind[is_leaves]

        values_sorted = dec_tree.tree_.__getstate__()['values']
        node_pred = np.argmax(values_sorted[:,0,:], axis=1)
        leaves_class_index = node_pred[is_leaves]

        ### TO BE SANITY-CHECKED
        node_unvisited = np.ones((n_classes, n_nodes), dtype=bool)
        tree_importances = np.zeros((n_classes, n_features))
        for leaf_i,leaf_c_i in zip(leaves_index,leaves_class_index):
            parent_i = parent_node_ind[leaf_i]
            current_i = leaf_i

            # walk the tree and calculate the importance of the predictor
            while parent_i != -1 and node_unvisited[leaf_c_i,current_i]:
                current_node = nodes[current_i]
                left_node = nodes[current_node['left_child']]
                right_node = nodes[current_node['right_child']]
                current_feature = current_node['feature']
                tree_importances[leaf_c_i,current_feature] += (
                        current_node['weighted_n_node_samples'] * current_node['impurity'] -
                        left_node['weighted_n_node_samples'] * left_node['impurity'] -
                        right_node['weighted_n_node_samples'] * right_node['impurity']
                        )
                node_unvisited[leaf_c_i,current_i] = False
                current_i = parent_i
                parent_i = parent_node_ind[current_i]
        importance_matrix.append(tree_importances)

    # average the predictor importances for each class by all of the trees in the forest
    importance_matrix = np.mean(importance_matrix, axis = 0)
    return(importance_matrix)