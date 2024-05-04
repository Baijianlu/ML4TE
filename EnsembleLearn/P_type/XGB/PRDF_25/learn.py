import numpy as np
import pandas as pd
from xgboost import XGBClassifier
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
    X_train = np.abs(df1.to_numpy())
    X_test = np.abs(df2.to_numpy())
    
    estimators = [('scaler', StandardScaler()), ('clf', XGBClassifier(tree_method='exact',importance_type='gain',scale_pos_weight=2.0954545,n_jobs=4,random_state=0))] #
    pipe = Pipeline(estimators)                     

    param_grid = {"clf__subsample": [0.5,0.6,0.7,0.8,0.9], "clf__colsample_bynode":[0.6,0.7,0.8,0.9,1.0], "clf__n_estimators": [100,150,200,250,300,400], "clf__learning_rate": [0.35,0.37,0.39,0.4,0.41,0.43,0.45], "clf__max_depth": [2,3,4,5,6]}
    scores = ['f1', 'precision', 'recall']

    cv = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=scores, refit='precision', cv=cv, n_jobs=2)

    grid_search.fit(X_train, Y_train)
    results_df = pd.DataFrame(grid_search.cv_results_)
    print(grid_search.best_params_)
    Y_pred = grid_search.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    results_df.to_csv('cv_results.csv', index_label='index', sep="\t")
    
    ###save the model
    pkl_model = "model.pkl"
    with open(pkl_model, 'wb') as pklM:
        pickle.dump(grid_search.best_estimator_, pklM)


    
