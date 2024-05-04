import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import pickle

if __name__ == '__main__':
    '''This is for training the model! '''
    df2 = pd.read_excel("test.xlsx", sheet_name='Sheet1',index_col=0)
    df3 = pd.read_excel("X_test_p.xlsx", sheet_name='Sheet1',index_col=False)
    Y_test = df2['class'].to_numpy()
    df2.drop(columns=['class'],inplace=True)
    X_test = np.abs(df2.to_numpy())

    with open('model.pkl','rb') as pklM:
        pkl_model = pickle.load(pklM)

    Y_pred = pkl_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    Y_score = pkl_model.predict_proba(X_test)[:, 1]
    AUC = roc_auc_score(Y_test, Y_score)
    print(f"AUC = {AUC:.2f}")
    df3['prob'] = pd.Series(Y_score, dtype=float)
    df3['pred'] = pd.Series(Y_pred, dtype=int)
    df3.to_excel('X_test_p_withpreds.xlsx', index_label='index', merge_cells=False)

    
