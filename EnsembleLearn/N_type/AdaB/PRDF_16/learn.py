import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
    
    dt_stump = DecisionTreeClassifier(min_weight_fraction_leaf=0.0025,class_weight='balanced')
    estimators = [('scaler', StandardScaler()), ('clf', AdaBoostClassifier(base_estimator=dt_stump, random_state=0))] #
    pipe = Pipeline(estimators)                     

    param_grid = {"clf__n_estimators": [200,300,400,500,600,650,700,750,800], "clf__learning_rate": [0.02,0.04,0.05,0.06,0.07,0.08,0.1], "clf__base_estimator__max_depth": [1,2,3]}
    scores = ['f1', 'precision', 'recall']

    cv = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=scores, refit='precision', cv=cv, n_jobs=4)

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


    
