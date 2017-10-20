import xgboost as xgb
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


train_feat=pd.read_csv('data/train.csv')
eval_feat=pd.read_csv('data/evaluation_public.csv')
predictors = list(train_feat.columns)
predictors.remove('id')
predictors.remove('label')
train_feat.loc[(train_feat['label']=='NEGATIVE'),'label']=0
train_feat.loc[(train_feat['label']=='POSITIVE'),'label']=1

params = {
    'objective': ['binary:logistic'],
    'eta': [0.3],
    'colsample_bytree': [1],
    'min_child_weight': [3],
    'max_depth': [4],
    'subsample': [0.8],
    'gamma': [0.1],
    'lambda':[0.1],
    'verbose_eval': [True],
    'silent':[1]
}
params = list(ParameterGrid(params))
for param in params:
    print(param)
    xgbtrain = xgb.DMatrix(train_feat[predictors],train_feat['label'],feature_names=predictors)
    model = xgb.train(param, xgbtrain, num_boost_round=140,verbose_eval=param['verbose_eval'])

#result
eval_feat.loc[:,'label']='NEGATIVE'
eval=eval_feat[(eval_feat[predictors[0]]!=-9999)]
xgbtest=xgb.DMatrix(eval[predictors],feature_names=predictors)
eval['pred']=model.predict(xgbtest)
eval=eval[['id','pred']]
eval_feat=pd.merge(eval_feat,eval,on='id',how='left')
eval_feat['pred'].fillna(0, inplace=True)
eval_feat.loc[(eval_feat['pred']>0.5),'label']='POSITIVE'
result=eval_feat[['id','label']]
result.to_csv('result.csv',index=False,header=False)


