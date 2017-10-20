import xgboost as xgb
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

train_feat=pd.read_csv('data/train.csv')
predictors = list(train_feat.columns)
predictors.remove('id')
predictors.remove('label')
train_feat.loc[(train_feat['label']=='NEGATIVE'),'label']=0
train_feat.loc[(train_feat['label']=='POSITIVE'),'label']=1

params = {
    'objective': ['binary:logistic'],
    'eta': [0.3],
    'colsample_bytree': [0.8],
    'min_child_weight': [3],
    'max_depth': [5],
    'subsample': [0.8],
    'gamma': [0.1],
    'lambda':[0.1],
    'verbose_eval': [True],
    'silent':[1],
    'scale_pos_weight':[1]
}
params = list(ParameterGrid(params))
for param in params:
    print(param)
    xgbtrain = xgb.DMatrix(train_feat[predictors],train_feat['label'],feature_names=predictors)
    # model = xgb.cv(param, xgbtrain, num_boost_round=100,nfold=5,verbose_eval=param['verbose_eval'])
    model = xgb.train(param, xgbtrain, num_boost_round=140,verbose_eval=param['verbose_eval'])
del train_feat,xgbtrain
gc.collect()

#feature importance
feature_importance=model.get_fscore()
feature_names = list(feature_importance.keys())
feature_importance=list(feature_importance.values())
y_pos = np.arange(len(feature_importance))
plt.barh(y_pos,feature_importance,align = 'center',alpha = 0.2,color='b')
plt.yticks(y_pos,feature_names)
plt.show()