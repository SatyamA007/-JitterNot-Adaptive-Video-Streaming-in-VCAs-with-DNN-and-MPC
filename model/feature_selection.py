# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from constants import *
import os

def feature_selection_selectKBest(X, y, k=6):
    # TRAIN_SPLIT, dataset = get_normalized_df(csv_path)
    # dataset = pd.DataFrame(dataset)
    # print(dataset.head())
    bestfeatures = SelectKBest(score_func=chi2, k=k)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print("Least important features are :")
    print(featureScores.nsmallest(k, 'Score'))
    print("Most important features are: ")
    print(featureScores.nlargest(k,'Score'))  #print 10 best features

def get_master_df():
    df_list = []
    ct = 100
    for dir in os.listdir(csv_dir):
        if os.path.isdir(os.path.join(csv_dir, dir)):
            for csv_file in os.listdir(os.path.join(csv_dir, dir)):
                print("*****************************************************")
                print(csv_file)
                if 'DS_Store' in csv_file:
                    continue
                ct -= 1
                csv_path = os.path.join(csv_dir, dir, csv_file)
                df = pd.read_csv(csv_path)
                df_list.append(df)
    master_df = pd.concat(df_list)
    master_df = master_df.assign(label=lambda x: x['outbound_packetsSent/s'])
    master_df.drop('outbound_packetsSent/s', axis=1, inplace=True)
    X = master_df.iloc[:, :-1]  # independent columns
    y = master_df.iloc[:, -1]
    return X, y

def feature_selection_ExtraTreesClassifier(X, y, k=6):
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(k).plot(kind='barh')
    plt.show()

def get_best_features():
    X, y = get_master_df()
    feature_selection_selectKBest(X, y)
    feature_selection_ExtraTreesClassifier(X, y)

get_best_features()