import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import pickle

if __name__ == '__main__':
    '''This is for training the model! '''
    df1 = pd.read_excel("train.xlsx", sheet_name='Sheet1',index_col=0)
    df2 = pd.read_excel("test.xlsx", sheet_name='Sheet1',index_col=0)
    Y_train = df1['class'].to_numpy()
    Y_test = df2['class'].to_numpy()
    df1.drop(columns=['class'],inplace=True)
    df2.drop(columns=['class'],inplace=True) 
    X_train = df1.to_numpy()
    X_test = df2.to_numpy()
 
    estimators = [('scaler', StandardScaler()), ('clf', GradientBoostingClassifier(subsample=0.5, min_weight_fraction_leaf=0.003, random_state=0))] #
    pipe = Pipeline(estimators)                     

    param_grid = {"clf__n_estimators": [100,200,300,400,450,500,550,600], "clf__learning_rate": [0.01,0.02,0.04,0.06,0.07,0.08,0.09], "clf__max_depth": [2,3,4,5,6]}
    scores = ['f1', 'precision', 'recall']

    cv = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=scores, refit='precision', cv=cv, n_jobs=4)

    sample_weight = np.zeros((len(Y_train),))
    for i in np.arange(len(Y_train)):
        if Y_train[i] == 0:
            sample_weight[i] = 0.84711779
        else:
            sample_weight[i] = 1.22021661
        
    grid_search.fit(X_train, Y_train, clf__sample_weight=sample_weight)
    results_df = pd.DataFrame(grid_search.cv_results_)
    print(grid_search.best_params_)
    Y_pred = grid_search.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    results_df.to_csv('cv_results.csv', index_label='index', sep="\t")
    
    ###save the model
    pkl_model = "model.pkl"
    with open(pkl_model, 'wb') as pklM:
        pickle.dump(grid_search.best_estimator_, pklM)


    
