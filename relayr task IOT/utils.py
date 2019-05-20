from __future__ import division
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import xgboost as xgb
from time import time
import operator

RANDOM_STATE=2017

xgb0 = {
    "objective": "reg:logistic",
    "learning_rate": 0.1,
    #"max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    'eval_metric':"auc"
}

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def addTfIdfFeats(df, data, exclude, tfidf, svd, clf, cols, param_name, colSet, clust=False):
    for col in cols:
        colname=col+'_clrd'
        df[colname] = df[col].apply(lambda x: ''.join(ch for ch in str(x) if ch not in exclude))
        XallIdf = tfidf.fit_transform(df[colname].fillna('empty')) 
        feature_names = tfidf.get_feature_names() 
        #print (col, np.shape(XallIdf)[1], feature_names[:10])
        X_svd = svd.fit_transform(XallIdf)
        X_scaled = StandardScaler().fit_transform(X_svd)
        if clust:
            X_tsne = clf.fit_predict(X_scaled) 
            data[param_name+col] = X_tsne
            colSet.append(param_name+col)
        else:
            X_tsne = clf.fit_transform(X_scaled)
            for i in range(np.shape(X_tsne)[1]):
                data[param_name+col+str(i)] = X_tsne[:, i]
                colSet.append(param_name+col+str(i))
    return data, colSet

def simple_xgb(paramz, data, y, skf, ESR, seed, breakAfterFold0=False, Verbose=False, show_feature_imp=False, OnlyMeanAUC=False, returnTrainPred=False):
    t0 = time()
    metric_list=[]
    n_rounds_list=[]
    train_len = len(y)
    train_pred = np.zeros(train_len)
    #print (data.columns)
    
    for i, (train_index, valid_index) in enumerate(skf):
        xtrain = data[:train_len].iloc[train_index]
        ytrain = y[train_index] 
        xtest = data[:train_len].iloc[valid_index]
        ytest = y[valid_index]
        

        dtrain = xgb.DMatrix(xtrain, label=ytrain)
        dvalid = xgb.DMatrix(xtest)
        deval = xgb.DMatrix(xtest, label=ytest)
        watchlist = [(dtrain, 'train'), (deval, 'eval')]
        
        paramz['random_state'] = seed#(i+1)+1
        paramz['seed'] = seed#(i+1)+1 
        paramz['silent']=1
        clf = xgb.train(paramz, dtrain, 10000, watchlist, early_stopping_rounds=ESR, verbose_eval=Verbose)            
        pred_fold = clf.predict(dvalid)
        fold_metric = roc_auc_score(y[valid_index], pred_fold)
        metric_list.append(fold_metric)
        
        train_pred[valid_index] = pred_fold
        n_rounds_list.append(clf.best_iteration)
        print ('Fold: %s AUC: %0.5f nrounds: %s time: %s min' % \
                (i, fold_metric, clf.best_iteration, np.round((time() - t0) / 60. ,1)))
        
        if breakAfterFold0:
            if not show_feature_imp:
                return fold_metric
            else:
                 #feature_imp    
                features = data.columns[:]  
                create_feature_map(features)
                importance = clf.get_score(fmap='xgb.fmap', importance_type='gain')
                importance = sorted(importance.items(), key=operator.itemgetter(1))
                fi = pd.DataFrame(importance, columns=['feature', 'fscore'])
                fi['fscore'] = fi['fscore'] / fi['fscore'].sum()
                fi.sort_values(by='fscore', ascending=False, inplace=True)
                print (fi[:50]) # top-TOPN 
                return fold_metric

    if returnTrainPred:
        return train_pred 
        
    if OnlyMeanAUC:
        return np.mean(metric_list)

    print ('Mean AUC: %0.5f (std: %0.5f) mean nrounds: %s' % (np.mean(metric_list), np.std(metric_list), np.mean(n_rounds_list)))

  
    if show_feature_imp:    
        #feature_imp 
        print ('-'*30)
        print ('Feature importance (gini):')   
        features = data.columns[:]  
        create_feature_map(features)
        importance = clf.get_score(fmap='xgb.fmap', importance_type='gain')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        fi = pd.DataFrame(importance, columns=['feature', 'fscore'])
        fi['fscore'] = fi['fscore'] / fi['fscore'].sum()
        fi.sort_values(by='fscore', ascending=False, inplace=True)
        print (fi[:]) # top-TOPN
        return fi.feature