
"""
   Usefull functions

__author__

    Alex Tselikov < atselikov@gmail.com >

"""

import pandas as pd
import numpy as np

#2date
def look_to_date(s):
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.apply(lambda v: dates[v])

#catEncoder
def catEncoder(series, nCats=10):
            return dict([(x, i) for i, x in enumerate(series.value_counts().index[:nCats])])

def PrepareSubset(DF, CatEncodingLim, ContTrasformType):
    #print 'unique types: ', np.unique(DF.dtypes)
    
    #object
    for col in DF.select_dtypes(include=['object']).columns:
        enc = catEncoder(DF[col], CatEncodingLim)
        DF[col] = DF[col].map(enc)    
    
    # non-object 
    for col in DF.select_dtypes(exclude=['object']).columns:
        DF[col] = ContTrasform(DF[col], ContTrasformType)
        
    return DF

def ContTrasform(Ser, ContTrasformType):  
    lim = 0.999
    if   ContTrasformType=='none':
        return Ser
    elif ContTrasformType=='log1p':
        return np.log1p(Ser)
    elif ContTrasformType=='power1/8':
        return np.power(Ser,1/8)
    elif ContTrasformType=='scale_sqrt':
        return np.sqrt(minmax_scale(Ser.fillna(-1)))
    elif ContTrasformType=='rank':
        values = rankdata(Ser).astype(np.float64)
        values = minmax_scale(values, feature_range=(-lim, lim))
        return scale(erfinv(values))
    else:
        print ('Error: unnown ContTrasformType!: ', ContTrasformType)

def GetSubset(DF, CorrLim, CoordLim, NullLim, SkewedLim, BirthVar, GenderVar):
    ulist=list(CorrLim + CoordLim + NullLim + SkewedLim + BirthVar + GenderVar)
    DF = SafeDrop(ulist, DF)
    return DF
    
def SafeDrop(ColList, DF):
    for col in ColList:
        if col in DF.columns:
            #if col != TARGET:
            DF = DF.drop(col,1)
    return DF        
        
#get FI xgb
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

#dyakonov  lin2classes    
def lin2classes_accuracy_dyak(y_test, pred, k):
    #return accuracy_score(y_test.astype(int), pred.astype(int))
    lres=[]
    gr=np.arange(1.0, 2.0, 0.01)
    for lk in gr:    
        lres.append(accuracy_score(y_test, np.round(lk*(pred-np.mean(pred))+np.mean(pred))))
    k = gr[list(lres).index(max(lres))]
    print (k)
    return accuracy_score(y_test, np.round(k*(pred-np.mean(pred))+np.mean(pred)))
    
def lin2classes_accuracy_dyak_k(y_test, pred, k):    
    return accuracy_score(y_test, np.round(k*(pred-np.mean(pred))+np.mean(pred)))    
    
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
    
def target_encode_std(train_series=None,
                  test_series=None,
                  target=None,
                  noise_level=0):
    assert len(train_series) == len(target)
    assert train_series.name == test_series.name

    temp = pd.concat([train_series, target], axis=1)
    # Compute target mean
    aggregated_values = temp.groupby(by=train_series.name)[target.name].agg(["mean", "count", np.std])
    total_std = np.std(target)
    aggregated_values["std"].fillna(total_std, inplace=True)

    # Compute smoothing
    smoothing_component = aggregated_values["count"] * total_std ** 2
    smoothing = smoothing_component / (aggregated_values["std"] ** 2 + smoothing_component)

    # Apply average function to all target data
    mean_total = target.mean()
    mean_values = mean_total * (1 - smoothing) + aggregated_values["mean"] * smoothing

    mean_values_dict = mean_values.rank(axis=0, method='first').to_dict()

    train_columns = train_series.replace(mean_values_dict).fillna(mean_total)
    test_columns = test_series.replace(mean_values_dict).fillna(mean_total)
    
    return add_noise(train_columns, noise_level), add_noise(test_columns, noise_level)