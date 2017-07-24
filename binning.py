'''
Created on May 23, 2016

@author: imayassi
'''
from sklearn import tree
import pandas as pd
import numpy as np

def bin(data, response):


    df=data
    # df = df.astype(float)

    # array=df.values
    y = df[response]
    x = df.drop([response], axis=1)
    Y = y.values
    X = x.values
    x=X[:,0:]
    y=Y


    # X = np.float32(X)
    # Y=data['ABANDONED_FLAG']
    # X=data.drop(['ABANDONED_FLAG'], inplace=True,axis=1, errors='ignore')
    x = np.float32(x)
    # print X

    # clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=2, min_samples_leaf=10)
    clf=tree.DecisionTreeRegressor(max_depth=2, min_samples_leaf=1)
    clf.fit(x, y)
    tree_leaf=clf.tree_.apply(x)
    nodes=[]
    nnodes=len(clf.tree_.children_left)
    l_node=[]
    for i in range(nnodes-1):
        if clf.tree_.children_left[i]!= -1:
            nodes.append(i)
    bins={}
    #GET THE ROOT NODE SPLIT AND ADD IT TO THE DICT
    a=['>', clf.tree_.threshold[0]]
    bins[clf.tree_.children_right[0]]= a
    b= ['<=',clf.tree_.threshold[0]]
    bins[clf.tree_.children_left[0]]=b
    right_split=clf.tree_.children_right[0]

    for m in reversed(nodes):

        if np.where(clf.tree_.children_left==m)[0]>=0 and m<right_split: #CHECK IF THIS LEAF BELONGS TO THE LEFT CHILD AND IF IT IS ON LEFT SPLIT OF THE LEFT CHILD OF THE TREE (i.e. LEFT LEFT)

            i=np.where(clf.tree_.children_left==m)[0] #INDEX THE LOCATION OF THE NODE m IN THE LEFT CHILD ARRAY
            left_leaf_value=clf.tree_.children_left[m]
            right_leaf_value=clf.tree_.children_right[m]
            c=['> ',clf.tree_.threshold[m],  ' < ' ,  clf.tree_.threshold[i[0]]]
            bins[right_leaf_value]=c
            d=['<=' , clf.tree_.threshold[m]]
            bins[left_leaf_value]=d


        elif np.where(clf.tree_.children_left==m)[0]>=0 and m>=right_split: #CHECK IF THIS LEAF BELONGS TO THE LEFT CHILD AND IF IT IS ON LEFT SPLIT OF THE RIGHT CHILD OF THE TREE (i.e. LEFT RIGHT)
            i=np.where(clf.tree_.children_left==m)[0] #INDEX THE LOCATION OF THE NODE m IN THE LEFT CHILD ARRAY
            left_leaf_value=clf.tree_.children_left[m]
            right_leaf_value=clf.tree_.children_right[m]
            e=['> ',clf.tree_.threshold[m], ' < ' ,  clf.tree_.threshold[i[0]]]
            bins[right_leaf_value]=e
            f= '<=' + repr(clf.tree_.threshold[m])
            bins[left_leaf_value]=f

        elif np.where(clf.tree_.children_right==m)[0]>=0 and m>=right_split*1.5: #CHECK IF THIS LEAF BELONGS TO THE RIGHT CHILD AND IF IT IS ON LEFT SPLIT OF THE RIGHT CHILD OF THE TREE (i.e. RIGHT RIGHT)

            i=np.where(clf.tree_.children_right==m)[0] #INDEX THE LOCATION OF THE NODE m IN THE RIGHT CHILD ARRAY
            left_leaf_value=clf.tree_.children_left[m]
            right_leaf_value=clf.tree_.children_right[m]
            g=['> ',clf.tree_.threshold[i[0]], ' < ' ,clf.tree_.threshold[m]]
            bins[left_leaf_value]=g
            h=['>=',clf.tree_.threshold[m]]
            bins[right_leaf_value]=h

        elif np.where(clf.tree_.children_right==m)[0]>=0 and m<right_split*1.5:#CHECK IF THIS LEAF BELONGS TO THE RIGHT CHILD AND IF IT IS ON RIGHT SPLIT OF THE LEFT CHILD OF THE TREE (i.e. RIGHT LEFT)

            i=np.where(clf.tree_.children_right==m)[0] #INDEX THE LOCATION OF THE NODE m IN THE RIGHT CHILD ARRAY
            left_leaf_value=clf.tree_.children_left[m]
            right_leaf_value=clf.tree_.children_right[m]
            i=['> ',clf.tree_.threshold[i[0]], ' <' ,clf.tree_.threshold[m]]
            bins[left_leaf_value]=i
            j=['>=', clf.tree_.threshold[m]]
            bins[right_leaf_value]=j
        print bins



    return bins




