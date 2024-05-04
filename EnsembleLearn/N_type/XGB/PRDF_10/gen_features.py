import numpy as np
import pandas as pd
from pymatgen.core import Structure
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
from matminer.featurizers.structure import PartialRadialDistributionFunction, StructureComposition, DensityFeatures
from matminer.featurizers.site import GeneralizedRadialDistributionFunction
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.conversions import DictToObject

if __name__ == '__main__':
    '''For sci-kit learning! '''
    "This part for generating features"
    df1 = pd.read_excel("X_train_n.xlsx", sheet_name='Sheet1',index_col=False)
    df2 = pd.read_excel("X_test_n.xlsx", sheet_name='Sheet1',index_col=False)
    train_structures=[Structure.from_str(struct, 'json') for struct in df1['structure_opt']]
    test_structures=[Structure.from_str(struct, 'json') for struct in df2['structure_opt']]
    df1['structure']=pd.Series(train_structures)
    df2['structure']=pd.Series(test_structures)
    prdf_feat = PartialRadialDistributionFunction(cutoff=20, bin_size=0.1)
    prdf_feat.fit(df1['structure'])
    featurizer = MultipleFeaturizer([
        StructureComposition(Stoichiometry()),
        StructureComposition(ElementProperty.from_preset("magpie")),
        StructureComposition(ValenceOrbital(props=['frac'])),
        StructureComposition(IonProperty(fast=True)),
        prdf_feat,
        DensityFeatures()
    ])
    PD_train = featurizer.featurize_dataframe(df1, "structure")
    PD_test = featurizer.featurize_dataframe(df2, "structure")

    excludes = ['index','name','structure_opt','n_con','n_PF','structure','compound possible']
    PD_train = PD_train.drop(columns=excludes)
    PD_test = PD_test.drop(columns=excludes)
    excludes = list()
    
    for col in PD_train.columns:
        if (np.std(PD_train[col]) < 1e-9) and (col not in excludes):
            excludes.append(col)
        
    PD_train = PD_train.drop(columns=excludes)
    PD_test = PD_test.drop(columns=excludes)
    PD_train.to_excel('train.xlsx', index_label='index', merge_cells=False)
    PD_test.to_excel('test.xlsx', index_label='index', merge_cells=False)

