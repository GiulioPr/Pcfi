#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:30:47 2018

@author: quanti
"""

###MERGE RAW DATA

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.ticker import FixedLocator, ScalarFormatter, FormatStrFormatter
import scipy.stats as sps
import copy as cp
import random as rnd
import itertools as itt
import seaborn as sns
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mpc
import matplotlib.patches as mpatches
import scipy.optimize as spo
import scipy.special as spsp
import pickle
import umap

import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics

files_path = '/0_data/rawer/'
proj_dir = os.path.dirname(os.path.dirname(os.getcwd()))
file_names_lst = os.listdir(proj_dir + files_path)
print('Should be %s:'%(64*2*3), len(file_names_lst))
df_lst = []
CDs = ['CD4', 'CD8']
stages = ['Control', '10day', '20day']
for fn in file_names_lst:
    df_pop = pd.read_csv(proj_dir + files_path + fn, sep='\t', header=1,
                         usecols=lambda x: not(x in ['Event #']))
                                                     
    CD_tag = [cd for cd in CDs if cd in fn][0]
    stage_tag = [stg for stg in stages if stg in fn][0]
    pop_tag = fn[fn.find('Population')+len('Population')+1:fn.find('.txt')]
    df_pop['CD'] = [CD_tag] * len(df_pop)
    df_pop['Stage'] = [stage_tag] * len(df_pop)
    df_pop['Population'] = [pop_tag] * len(df_pop)
    df_lst.append(df_pop)
df_pool = pd.concat(df_lst, ignore_index=True)
"""
for fn in file_names_lst:
    df_pop = pd.read_csv(proj_dir + files_path + fn, sep='\t', header=1,
                         usecols=lambda x: not(x in ['Event #']))
    pop_tag = int(fn[fn.find('Population_')+len('Population_'):fn.find('_Time_')])
    df_pop['Population'] = np.ones(len(df_pop), dtype=int)*pop_tag
    df_lst.append(df_pop)
df_CD4 = pd.concat(df_lst, ignore_index=True)

#CD8
files_path = '/0_data/raw/CD8_files/'
proj_dir = os.path.dirname(os.path.dirname(os.getcwd()))
file_names_lst = os.listdir(proj_dir + files_path)
print('Should be 64:', len(file_names_lst))
df_lst = []
for fn in file_names_lst:
    df_pop = pd.read_csv(proj_dir + files_path + fn, sep='\t', header=1,
                         usecols=lambda x: not(x in ['Event #']))
    pop_tag = int(fn[fn.find('Population_')+len('Population_'):fn.find('_Time_')])
    df_pop['Population'] = np.ones(len(df_pop), dtype=int)*pop_tag
    df_lst.append(df_pop)
df_CD8 = pd.concat(df_lst, ignore_index=True)
#Collection of features 
panel_features = ['hCD45', 'CD90.2', 'CD45.2', 'Ter119', 'CD11b', 'Ly6GC',
       'pCDK1_cdc2', 'CD69', 'pBRCA1', 'CD4', 'pATM', 'pH2AX', 'CyclinB1',
       'KLRG1', 'CD27', 'Ki67', 'CD3', 'CD45.1', 'TIM3', 'OX40', 'RGS1',
       'pRad51', 'Foxp3', 'PD1', 'tbet', 'pATR', 'p21', 'pBRCA2', 'CD62L',
       'NK1.1', 'CD19', 'Rad51', 'CD8', 'TCRb', 'CD137', 'CD44', 'CD86',
       'CTLA4', 'CD223', 'pHH3', 'B220', 'MHCII']
#Eliminated: ['Time', 'Event_length', 'BC1', 'BC2', 'BC3', 'BC4', 'BC5','BC6',
#'I127Di', 'Pt195Di', 'DNA1', 'DNA2',
#'beadDist', 'bc_separation_dist', 'mahalanobis_dist', 'Population' 
pop6_features = ['OX40', 'CD137', 'PD1', 'TIM3', 'CTLA4', 'CD223']
res_panel_features = [#LISTED IN PRELIMINARY PLOTS
        'pHH3',  'CD86', 'CD62L', 'CD90.2', 'pBRCA2', 'CD69', 
        'pRad51', 'p21', 'Foxp3', 'MHCII', 'CyclinB1', 'CD27',
        'tbet', 'CD44', 'KLRG1', 'TCRb', 'pCDK1_cdc2', 'CD3',
        'Ly6GC', 'pBRCA1', 'pH2AX', 'pATR', 'Rad51',
        #UNLISTED IN PRELIMINARY PLOTS
        'hCD45', 'CD45.2', 'CD11b', 'CD45.1', 'RGS1',
        'Ki67', 'pATM', 'B220',
        #MAILY FOR GATING
        'NK1.1', 'CD19', 'CD8', 'CD4', 'Ter119']
"""
#Collection of features 
panel_features = ['hCD45_(Y89Di)', 'CD90.2_(In113Di)', 'CD45.2_(In115Di)',
       'Ter119_(La139Di)', 'CD11b_(Ce140Di)', 'Ly6GC_(Pr141Di)',
       'pCDK1_cdc2_(Nd142Di)', 'CD69_(Nd143Di)', 'pBRCA1_(Nd144Di)',
       'CD4_(Nd145Di)', 'pATM_(Nd146Di)', 'pH2AX_(Pm147Di)',
       'CyclinB1_(Nd148Di)', 'KLRG1_(Sm149Di)', 'CD27_(Sm150Di)',
       'Ki67_(Eu151Di)', 'CD3_(Sm152Di)', 'CD45.1_(Eu153Di)', 'TIM3_(Sm154Di)',
       'OX40_(Gd155Di)', 'RGS1_(Gd156Di)', 'pRad51_(Gd157Di)',
       'Foxp3_(Gd158Di)', 'PD1_(Tb159Di)', 'tbet_(Gd160Di)', 'pATR_(Dy161Di)',
       'p21_(Dy162Di)', 'pBRCA2_(Dy163Di)', 'CD62L_(Dy164Di)',
       'NK1.1_(Ho165Di)', 'CD19_(Er166Di)', 'Rad51_(Er167Di)', 'CD8_(Er168Di)',
       'TCRb_(Tm169Di)', 'CD137_(Er170Di)', 'CD44_(Yb171Di)', 'CD86_(Yb172Di)',
       'CTLA4_(Yb173Di)', 'CD223_(Yb174Di)', 'pHH3_(Lu175Di)',
       'B220_(Yb176Di)',
       'MHCII_(Bi209Di)']
"""panel_features = ['hCD45', 'CD90.2', 'CD45.2', 'Ter119', 'CD11b', 'Ly6GC',
       'pCDK1_cdc2', 'CD69', 'pBRCA1', 'CD4', 'pATM', 'pH2AX', 'CyclinB1',
       'KLRG1', 'CD27', 'Ki67', 'CD3', 'CD45.1', 'TIM3', 'OX40', 'RGS1',
       'pRad51', 'Foxp3', 'PD1', 'tbet', 'pATR', 'p21', 'pBRCA2', 'CD62L',
       'NK1.1', 'CD19', 'Rad51', 'CD8', 'TCRb', 'CD137', 'CD44', 'CD86',
       'CTLA4', 'CD223', 'pHH3', 'B220', 'MHCII']"""
#Eliminated: ['Time', 'Event_length', 'BC1_(Pd102Di)', 'BC2_(Pd104Di)',
        #'BC3_(Pd105Di)', 'BC4_(Pd106Di)', 'BC5_(Pd108Di)', 'BC6_(Pd110Di)',
        #'I127Di', 'Pt195Di', 'DNA1_(Ir191Di)', 'DNA2_(Ir193Di)',
        #'beadDist', 'bc_separation_dist', 'mahalanobis_dist',
        #'File Number_(FileNum)', 'CD', 'Stage', 'Population']
pop6_features = ['OX40_(Gd155Di)', 'CD137_(Er170Di)', 'PD1_(Tb159Di)',
                 'TIM3_(Sm154Di)', 'CTLA4_(Yb173Di)', 'CD223_(Yb174Di)']
res_panel_features = [
        #LISTED IN PRELIMINARY PLOTS
        'CD90.2_(In113Di)', 'Ly6GC_(Pr141Di)', 'pCDK1_cdc2_(Nd142Di)',
        'CD69_(Nd143Di)', 'pBRCA1_(Nd144Di)', 'pH2AX_(Pm147Di)',
        'CyclinB1_(Nd148Di)', 'KLRG1_(Sm149Di)', 'CD27_(Sm150Di)',
        'CD3_(Sm152Di)', 'pRad51_(Gd157Di)', 'Foxp3_(Gd158Di)',
        'tbet_(Gd160Di)', 'pATR_(Dy161Di)', 'p21_(Dy162Di)',
        'pBRCA2_(Dy163Di)', 'CD62L_(Dy164Di)', 'Rad51_(Er167Di)', 
        'TCRb_(Tm169Di)', 'CD44_(Yb171Di)', 'CD86_(Yb172Di)',
        'pHH3_(Lu175Di)', 'MHCII_(Bi209Di)',
        #UNLISTED IN PRELIMINARY PLOTS
        'hCD45_(Y89Di)', 'CD45.2_(In115Di)', 'CD11b_(Ce140Di)',
        'CD45.1_(Eu153Di)', 'RGS1_(Gd156Di)', 'Ki67_(Eu151Di)',
        'pATM_(Nd146Di)', 'B220_(Yb176Di)', 
        #MAINLY FOR GATING
        'CD4_(Nd145Di)', 'CD8_(Er168Di)', 'NK1.1_(Ho165Di)',
        'CD19_(Er166Di)', 'Ter119_(La139Di)', 
        ]
stop
#CHECK
#'Ter119', metal?
##SHOULD EXCLUDE THE MARKERS FOR GATING!


#NOTES: should eliminate unobserved population
#should 
#PLAY
df_CD8 = cp.deepcopy(df_pool[df_pool.CD=='CD8'])
df_CD8.loc[:,panel_features] = np.arcsinh(df_CD8.loc[:,panel_features]/5.)
pops, pops_count = np.unique(df_CD8.Population, return_counts=True)
df_CD8 = df_CD8[df_CD8.Population.isin(pops[pops_count>=10])]
X = df_CD8.loc[:,res_panel_features]
y = df_CD8.loc[:,['Population']]
# Split dataset into training set and test set
is_split_good = [False]
while all(is_split_good):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    is_split_good = np.array([k in np.unique(y_train) for k in np.unique(y_test)])
    print(np.unique(y_test)[~(is_split_good)])

et_clf = ExtraTreesClassifier(n_estimators=300,# max_features=None,
                              random_state=0)
et_clf.fit(X_train, np.ravel(y_train))
y_pred_et = et_clf.predict(X_test)
feature_imp = pd.Series(et_clf.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
print(feature_imp)
print(metrics.confusion_matrix(y_test,y_pred_et))  
print(metrics.accuracy_score(y_test, y_pred_et))#->0.388



rf_clf = RandomForestClassifier(n_estimators=300,
                                #max_features=int(np.ceil(np.sqrt(len(res_panel_features)))),
                                #max_features=len(res_panel_features),
                                random_state=0)
rf_clf.fit(X_train, np.ravel(y_train))
y_pred_rf = rf_clf.predict(X_test)
feature_imp = pd.Series(rf_clf.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
print(feature_imp)
print(metrics.confusion_matrix(y_test,y_pred_rf))
print(metrics.accuracy_score(y_test, y_pred_rf))#->0.392
conf_mat = metrics.confusion_matrix(y_test,y_pred_rf)
print(metrics.classification_report(y_test, y_pred_rf))##, target_names=target_names))
perf_dct = metrics.classification_report(y_test, y_pred_rf,output_dict=True)
stop
###TO BE SANITY-CHECKED
"""
node_unvisited = np.ones((len(classes), n_nodes), dtype=bool)
tree_importances = np.zeros((len(classes),len(res_panel_features)))
k = 0
for leaf_i,leaf_c_i in zip(leaves_index,leaves_class_index):
    parent_i = parent_node_ind[leaf_i]
    current_i = leaf_i

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
#all([node[0]==-1 for node in leaves])
#all([node[1]==-1 for node in leaves])
#all([node[2]==-2 for node in leaves])
#all([node[3]==-2 for node in leaves])
#all([node[4]==0 for node in leaves])

tree_importances_norm_class = tree_importances/np.sum(tree_importances, axis=0)
tree_importances_norm_tot = tree_importances/np.sum(tree_importances)
tree_importances_norm_feat = tree_importances.T/np.sum(tree_importances.T, axis=0)

df_tree = pd.DataFrame(data=tree_importances_norm_class,
                       index=classes, columns=res_panel_features)
fig,ax = plt.subplots()
sns.heatmap(df_tree, ax=ax)
fig.show()

df_tree = pd.DataFrame(data=tree_importances_norm_feat.T,
                       index=classes, columns=res_panel_features)
fig,ax = plt.subplots()
sns.heatmap(df_tree, ax=ax)
fig.show()

df_tree = pd.DataFrame(data=tree_importances_norm_tot,
                       index=classes, columns=res_panel_features)
rnk_cols = sorted(list(df_tree.columns.values), key=lambda x:df_tree[x].sum())
rnk_rows = sorted(list(df_tree.index.values), key=lambda x:df_tree.loc[x,:].sum())
df_reord = df_tree.loc[rnk_rows,rnk_cols]
fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_reord, ax=ax, xticklabels=True, yticklabels=True)
#ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=45)
sns.despine(fig=fig)
fig.show()

feature = dec_tree.tree_.feature

imp = dec_tree.tree_.compute_feature_importances(normalize=True)
imp_dec_tree = pd.Series(imp, index=X.columns).sort_values(ascending=False)
print(imp_dec_tree)
#print(metrics.confusion_matrix(y_test,y_pred))  
#print(metrics.classification_report(y_test,y_pred))  
#print(metrics.accuracy_score(y_test, y_pred))  
"""

#HEATMAP
df_CD8 = cp.deepcopy(df_pool[df_pool.CD=='CD8'])
df_CD8.loc[:,panel_features] = np.arcsinh(df_CD8.loc[:,panel_features]/5.)
pops, pops_count = np.unique(df_CD8.Population, return_counts=True)
df_CD8 = df_CD8[df_CD8.Population.isin(pops[pops_count>=10])]

X = df_CD8.loc[:,res_panel_features]
y = df_CD8.loc[:,['Population']]
# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#VA SPLITTATO MEGLIO! PRENDI UN TERZO DA OGNI STAGE
rf_clf = RandomForestClassifier(n_estimators=300,
                                #max_features=int(np.ceil(np.sqrt(len(res_panel_features)))),
                                #max_features=len(res_panel_features),
                                random_state=0)

#et_clf = ExtraTreesClassifier(n_estimators=100, max_features=None,
#                              random_state=0)

rf_clf.fit(X, np.ravel(y))
#y_pred = rf_clf.predict(X_test)

feature_imp = pd.Series(rf_clf.feature_importances_,
                        index=X.columns).sort_values(ascending=False)
print(feature_imp)

#rf_clf.estimators_[0].tree_.impurity
importance_matrix = []
k = 0
for dec_tree in rf_clf.estimators_:
    k += 1
    #print(k)
    print(dec_tree.tree_.__getstate__()['max_depth'])
    #dec_tree = rf_clf.estimators_[0]
    classes = rf_clf.classes_#np.unique(y)
    #print('CHECK:', dec_tree.tree_.n_classes[0]==len(classes))
    feature = dec_tree.tree_.feature
    
    n_nodes = dec_tree.tree_.__getstate__()['node_count']
    nodes = dec_tree.tree_.__getstate__()['nodes']
    parent_node_ind = -np.ones(shape=n_nodes, dtype='<i8')
    for par_ind,node in enumerate(nodes):
        parent_node_ind[node[0]] = par_ind
        parent_node_ind[node[1]] = par_ind
    
    is_leaves = np.array([node[0]==node[1] for node in nodes])# np.zeros(shape=n_nodes, dtype=bool)
    #leaves = nodes[is_leaves]
    leaves_index = np.nonzero(is_leaves)[0]
    
    leaves_parent = parent_node_ind[is_leaves]
    
    values_sorted = dec_tree.tree_.__getstate__()['values']
    node_pred = np.argmax(values_sorted[:,0,:], axis=1)
    leaves_class_index = node_pred[is_leaves]
    #leaves_class = classes[leaves_class_index]
    
    
    ###TO BE SANITY-CHECKED
    node_unvisited = np.ones((len(classes), n_nodes), dtype=bool)
    tree_importances = np.zeros((len(classes),len(res_panel_features)))
    for leaf_i,leaf_c_i in zip(leaves_index,leaves_class_index):
        parent_i = parent_node_ind[leaf_i]
        current_i = leaf_i
    
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
#with open('./pickled_importance_matrices', 'wb') as fp:
#    pickle.dump(importance_matrix, fp)
with open('./pickled_importance_matrices', 'rb') as fp:
    importance_matrix = pickle.load(fp)
importance_matrix_ary = np.array(importance_matrix)
importance_matrix_ary = np.mean(importance_matrix_ary, axis=0)


df_tree_unnorm = pd.DataFrame(data=importance_matrix_ary,
                       index=classes, columns=res_panel_features)
rnk_cols_unnorm = sorted(list(df_tree_unnorm.columns.values),
                         key=lambda x:df_tree_unnorm[x].sum())
rnk_rows_unnorm = sorted(list(df_tree_unnorm.index.values),
                         key=lambda x:df_tree_unnorm.loc[x,:].sum())

importance_matrix_norm = importance_matrix_ary/np.sum(importance_matrix_ary, axis=0)
df_tree = pd.DataFrame(data=importance_matrix_norm,
                       index=classes, columns=res_panel_features)
rnk_rows = sorted(list(df_tree.index.values), key=lambda x:df_tree.loc[x,:].sum())
df_reord = df_tree.loc[rnk_rows,rnk_cols_unnorm]
fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_reord, ax=ax, xticklabels=True, yticklabels=True)
sns.despine(fig=fig)
fig.show()
fig.savefig('./Heatmap_renorm_prop_over_class.pdf', bbox_inches='tight')

importance_matrix_norm = importance_matrix_ary.T/np.sum(importance_matrix_ary.T, axis=0)
df_tree = pd.DataFrame(data=importance_matrix_norm.T,
                       index=classes, columns=res_panel_features)
rnk_cols = sorted(list(df_tree.columns.values), key=lambda x:df_tree[x].sum())
#rnk_rows = sorted(list(df_tree.index.values), key=lambda x:df_tree.loc[x,:].sum())
df_reord = df_tree.loc[rnk_rows_unnorm,rnk_cols]
fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_reord, ax=ax, xticklabels=True, yticklabels=True)
sns.despine(fig=fig)
fig.show()
fig.savefig('./Heatmap_renorm_prop_over_feat.pdf', bbox_inches='tight')

importance_matrix_norm = importance_matrix_ary/np.sum(importance_matrix_ary)
df_tree = pd.DataFrame(data=importance_matrix_norm,
                       index=classes, columns=res_panel_features)
#rnk_cols = sorted(list(df_tree.columns.values), key=lambda x:df_tree[x].sum())
#rnk_rows = sorted(list(df_tree.index.values), key=lambda x:df_tree.loc[x,:].sum())
df_reord = df_tree.loc[rnk_rows_unnorm,rnk_cols_unnorm]
fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(df_reord, ax=ax, xticklabels=True, yticklabels=True)
sns.despine(fig=fig)
fig.show()
fig.savefig('./Heatmap_renorm_tot.pdf', bbox_inches='tight')

pops, pops_count = np.unique(df_CD8.Population, return_counts=True)
count_reord = np.hstack([pops_count[pops==lab] for lab in rnk_rows_unnorm])
fig,ax = plt.subplots(figsize=(10,10))
sns.countplot(y='Population', data=df_CD8, order=rnk_rows_unnorm)
ax.invert_xaxis()
sns.despine(fig=fig)
#ax.set(xscale="log")
fig.show()
fig.savefig('./CD8_pop_size.pdf', bbox_inches='tight')


piano = np.array(list(itt.product([0,1], repeat=len(pop6_features))))
for k in np.arange(len(pop6_features), dtype=int)[::-1]:
    piano = sorted(piano, key=lambda row: row[k], reverse=True)
piano = np.array(sorted(piano, key=lambda row: row.sum()))
fig,ax = plt.subplots(figsize=(1.2,10))
sns.heatmap(piano, ax=ax, xticklabels=True, yticklabels=True, cbar=False,
            linecolor='Black', linewidth=0.1)
ax.set_xticklabels(pop6_features, rotation=90)
ax.set_yticklabels(np.arange(1,65))
sns.despine(fig=fig, bottom=True, top=False)
fig.show()
fig.savefig('./Piano.pdf', bbox_inches='tight')
fig,ax = plt.subplots(figsize=(15,1.5))
sns.heatmap(piano.T, ax=ax, xticklabels=True, yticklabels=True, cbar=False,
            linecolor='Black', linewidth=0.1)
ax.set_yticklabels(pop6_features, rotation=0)
ax.set_xticklabels(np.arange(1,65))
sns.despine(fig=fig, bottom=True, top=False)
fig.show()
fig.savefig('./Piano_hor.pdf', bbox_inches='tight')
#SORTED PIANO
sorted_piano = np.array([piano[int(k)-1] for k in rnk_rows_unnorm])
fig,ax = plt.subplots(figsize=(1.2,10))
sns.heatmap(sorted_piano, ax=ax, xticklabels=True, yticklabels=True, cbar=False,
            linecolor='Black', linewidth=0.1)
ax.set_xticklabels(pop6_features, rotation=90)
ax.set_yticklabels(rnk_rows_unnorm)
sns.despine(fig=fig, bottom=True, top=False)
fig.show()
fig.savefig('./Piano_sorted.pdf', bbox_inches='tight')


#UMAP
sns.set_style('darkgrid')

df_dct = {cd:cp.deepcopy(df_pool[df_pool.CD==cd]) for cd in CDs}
#dft, dfi, dfb = dct_data['pooled']
for cd in CDs:
    df = cp.deepcopy(df_dct[cd])
    print('Start UMAP', cd)
    embedding = umap.UMAP(#metric='cosine'#,
                          #n_neighbors=n_neighbors,
                          #min_dist=0.2
                          #n_components=n_components
                          ).fit_transform(
            np.arcsinh(df.loc[:,res_panel_features]/5.))
    print('End UMAP', cd)
    sub_indexes = np.random.choice(len(embedding), replace=False, size=100000)
    with open('./umap'+cd, 'wb') as fp:
        pickle.dump((embedding,sub_indexes), fp)
    with open('./umap'+cd, 'rb') as fp:
        (embedding,sub_indexes) = pickle.load(fp)
    #plt.scatter(embedding[:,1], embedding[:,0], s=1. ,c=cl_cols)
    for lvl in res_panel_features:
        fig,ax = plt.subplots()
        ax.scatter(embedding[sub_indexes,1], embedding[sub_indexes,0], s=0.1,
                   c=df[lvl].iloc[sub_indexes], cmap='viridis', alpha=0.2)
        #ax.scatter(embedding[:,1], embedding[:,0], s=1, c=df[lvl],
        #           cmap='viridis', alpha=0.2)
        sns.despine()
        #plt.xlim(xmin=-5)
        #ax.set_xlabels('')
        #ax.set_xticklabels('')
        #ax.set_yticklabels('')
        ax.set_title('UMAP '+cd+' '+lvl)
        fig.savefig('./Umap '+cd+' '+lvl+'.pdf', bbox_inches='tight')
        fig.show()
    for lvl in pop6_features:
        fig,ax = plt.subplots()
        ax.scatter(embedding[sub_indexes,1], embedding[sub_indexes,0], s=1,
                   c=df[lvl].iloc[sub_indexes], cmap='viridis', alpha=0.2)
        sns.despine()
        #plt.xlim(xmin=-5)
        #ax.set_xlabels('')
        #ax.set_xticklabels('')
        #ax.set_yticklabels('')
        ax.set_title('UMAP '+cd+' '+lvl)
        fig.savefig('./Umap '+cd+' '+lvl+'.pdf', bbox_inches='tight')
        fig.show()
    stage_col = {'Control':'blue', '10day':'gold', '20day':'red'}
    fig,ax = plt.subplots()
    ax.scatter(embedding[sub_indexes,1], embedding[sub_indexes,0], s=1,
               c=[stage_col[k] for k in df.Stage.iloc[sub_indexes]])
    sns.despine()
    #plt.xlim(xmin=-5)
    #ax.set_xlabels('')
    #ax.set_xticklabels('')
    #ax.set_yticklabels('')
    ax.set_title('UMAP '+cd+' tumor stages')
    fig.savefig('./Umap '+cd+' tumor stages.pdf', bbox_inches='tight')
    fig.show()
    pop_col = np.array([sns.cubehelix_palette(65)[int(k)] for k in df.Population])
    fig,ax = plt.subplots()
    ax.scatter(embedding[sub_indexes,1], embedding[sub_indexes,0], s=1,
               c=pop_col[sub_indexes], alpha=0.2)
    sns.despine()
    #plt.xlim(xmin=-5)
    #ax.set_xlabels('')
    #ax.set_xticklabels('')
    #ax.set_yticklabels('')
    ax.set_title('UMAP '+cd+' populations')
    fig.savefig('./Umap '+cd+' populations.pdf', bbox_inches='tight')
    fig.show()

