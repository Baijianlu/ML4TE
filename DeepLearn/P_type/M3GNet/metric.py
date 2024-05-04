import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

if __name__ == '__main__':
    df1 = pd.read_excel("X_test_p_withpreds.xlsx", sheet_name='Sheet1',index_col=0)
    Y_label = df1['class'].to_numpy()
    Y_pred = df1['pred'].to_numpy()
    Y_prob = df1['prob'].to_numpy()
    
    print(classification_report(Y_label, Y_pred))
   
    AUC = roc_auc_score(Y_label, Y_prob)
    print(f"AUC = {AUC:.2f}")
    
    
