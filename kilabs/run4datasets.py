
# coding: utf-8

"""
   Create different datasets

__author__

    Alex Tselikov < atselikov@gmail.com >

"""


from utils_tanz import *
from ats_functions import *
from ml_params import *
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing


if __name__ == "__main__":
    
    df, test, train, y, skf = load_data(dfold)
    
    
    cleaned_df = SetUncommonCatsToNan(df, test, train)
    
    
    #param for 0
    data = makeDataset(df, 'data0',
        amount_tsh='None', #log1p, quant5, quant9
        gps_height='None', #log1p, LatLong, Regions
        LatLong=['None'], #Correct, Regions,  r3, r2, r1, r0, ro ex: [Correct, Regions, r3]
        num_private='drop', # log1p, #Is0 None?
        population='log1p', #LatLong, Regions, None?
        construction_year='log1p', #LatLong, Regions
        date_recorded = 'drop', #proc
        wpt_name = 'drop', #LetCount, WordCount, LetWordRatio
        subvillage = 'drop', #LetCount, WordCount, LetWordRatio
        scheme_name = 'drop', #LetCount, WordCount, LetWordRatio
        installer = 'drop', #LetCount, WordCount, LetWordRatio
        funder = 'drop', #LetCount, WordCount, LetWordRatio
        ward = 'drop', #LetCount, WordCount, LetWordRatio
        lga = 'drop', #LetCount, WordCount
        extraction = 'None', #type, group, class, all
        management = 'scheme_management', #'management', management_group leave 1
        source='source', #source_type
        waterpoint_type='waterpoint_type', #waterpoint_type_group, 
        payment='payment', #payment_type
        quality='water_quality',#quality_group
        BoolFields='Nan2False' #ByGeo
            )
    
    #param for 1
    data = makeDataset(df, 'data1',
        amount_tsh='log1p', #log1p, quant5, quant9
        gps_height='log1p', #log1p, LatLong, Regions
        LatLong=['Correct', 'Regions',  'r3'], #Correct, Regions,  r3, r2, r1, r0, ro ex: [Correct, Regions, r3]
        num_private='log1p', # log1p, #Is0 None?
        population='LatLong', #LatLong, Regions, None?
        construction_year='LatLong', #LatLong, Regions
        date_recorded = 'proc', #proc
        wpt_name = 'LetCount', #LetCount, WordCount, LetWordRatio
        subvillage = 'LetCount', #LetCount, WordCount, LetWordRatio
        scheme_name = 'LetCount', #LetCount, WordCount, LetWordRatio
        installer = 'LetCount', #LetCount, WordCount, LetWordRatio
        funder = 'LetCount', #LetCount, WordCount, LetWordRatio
        ward = 'LetCount', #LetCount, WordCount, LetWordRatio
        lga = 'LetCount', #LetCount, WordCount
        extraction = 'type', #type, group, class, all
        management = 'management', #'management', management_group leave 1
        source='source_type', #source_type
        waterpoint_type='waterpoint_type', #waterpoint_type_group, 
        payment='payment', #payment_type
        quality='water_quality',#quality_group
        BoolFields='Nan2False' #ByGeo
            )
    
    
    #param2
    data = makeDataset(df, 'data2',
        amount_tsh='quant5', #log1p, quant5, quant9
        gps_height='LatLong', #log1p, LatLong, Regions
        LatLong=['Correct', 'Regions',  'r2'], #Correct, Regions,  r3, r2, r1, r0, ro ex: [Correct, Regions, r3]
        num_private='Is0', # log1p, #Is0 None?
        population='Regions', #LatLong, Regions, None?
        construction_year='Regions', #LatLong, Regions
        date_recorded = 'proc', #proc
        wpt_name = 'WordCount', #LetCount, WordCount, LetWordRatio
        subvillage = 'WordCount', #LetCount, WordCount, LetWordRatio
        scheme_name = 'WordCount', #LetCount, WordCount, LetWordRatio
        installer = 'WordCount', #LetCount, WordCount, LetWordRatio
        funder = 'WordCount', #LetCount, WordCount, LetWordRatio
        ward = 'WordCount', #LetCount, WordCount, LetWordRatio
        lga = 'WordCount', #LetCount, WordCount
        extraction = 'group', #type, group, class, all
        management = 'management_group', #'management', management_group leave 1
        source='source_type', #source_type
        waterpoint_type='waterpoint_type_group', #waterpoint_type_group, 
        payment='payment_type', #payment_type
        quality='quality_group',#quality_group
        BoolFields='ByGeo' #ByGeo
            )
    
    
    #param3
    data = makeDataset(df, 'data3',
        amount_tsh='quant9', #log1p, quant5, quant9
        gps_height='Regions', #log1p, LatLong, Regions
        LatLong=['Correct', 'Regions',  'r1'], #Correct, Regions,  r3, r2, r1, r0, ro ex: [Correct, Regions, r3]
        num_private='Is0', # log1p, #Is0 None?
        population='Regions', #LatLong, Regions, None?
        construction_year='Regions', #LatLong, Regions
        date_recorded = 'proc', #proc
        wpt_name = 'LetWordRatio', #LetCount, WordCount, LetWordRatio
        subvillage = 'LetWordRatio', #LetCount, WordCount, LetWordRatio
        scheme_name = 'LetWordRatio', #LetCount, WordCount, LetWordRatio
        installer = 'LetWordRatio', #LetCount, WordCount, LetWordRatio
        funder = 'LetWordRatio', #LetCount, WordCount, LetWordRatio
        ward = 'LetWordRatio', #LetCount, WordCount, LetWordRatio
        lga = 'LetWordRatio', #LetCount, WordCount
        extraction = 'all', #type, group, class, all
        management = 'management_group', #'management', management_group leave 1
        source='source_type', #source_type
        waterpoint_type='waterpoint_type_group', #waterpoint_type_group, 
        payment='payment_type', #payment_type
        quality='quality_group',#quality_group
        BoolFields='ByGeo' #ByGeo
            )
    
    
    # all variants
    dataall = makeDatasetAllVars(df.drop('status_group',1))
    dataall.to_csv('datasets/dataall.csv',index=False)
    
    
    # ## Run models
    
    le = preprocessing.LabelEncoder()
    
    
    #XGBcl label  data0clnd
    dataset, param_type ='data0cld', '_le_mc_params1'
    data, y, df, train, test = loadDataset(dataset)
    lcols=data.select_dtypes(include=['object']).columns
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])
    train_pred, test_pred, cur_accuracy, top60feats = run_mc_xgb(xgb_params_mc1, data, len(train), y, skf, 40)
    save_level1(train_pred, test_pred, cur_accuracy+'xgb_'+dataset+param_type)
    
    
    
    #XGBcl onehot data1clnd
    dataset, param_type ='data1cld', '_oh_mc_params2'
    data, y, df, train, test = loadDataset(dataset)
    lcols=data.select_dtypes(include=['object']).columns
    data=cats2oneHot(data,lcols)
    train_pred, test_pred, cur_accuracy, top60feats = run_mc_xgb(xgb_params_mc2, data, len(train), y, skf, 60)
    save_level1(train_pred, test_pred, cur_accuracy+'xgb_'+dataset+param_type)
    
    #XGBln + valuec data2
    dataset, param_type ='data2', '_vc_lin_params3'
    data, y, df, train, test = loadDataset(dataset)
    base=[xgb_params_lin3, [], [],[],[], [],  [],  30, 'none']
    train_pred, test_pred, cur_accuracy, top60feats = run_lin_xgb(base, data, len(train), y, skf, 30,10)
    save_level1(train_pred, test_pred, cur_accuracy+'xgb_'+dataset+param_type)    
    
    #XGBln + valuec data3
    dataset, param_type ='data3', '_vc_lin_params4'
    data, y, df, train, test = loadDataset(dataset)
    base=[xgb_params_lin4, [], [],[],[], [],  [],  60, 'none']
    train_pred, test_pred, cur_accuracy, top60feats = run_lin_xgb(base, data, len(train), y, skf, 60,10)
    save_level1(train_pred, test_pred, cur_accuracy+'xgb_'+dataset+param_type) 
    
    
    #SVR
    train_pred, test_pred, cur_accuracy = run_linear(LinearSVR(random_state=2017), pd.DataFrame(data), y, skf)
    save_level1(train_pred, test_pred, cur_accuracy+'svr_'+dataset+param_type)
    
    
    # #### rf on vc data2
    dataset, param_type ='data2', '_vc'
    data, y, df, train, test = loadDataset(dataset)
    lcols=data.select_dtypes(include=['object']).columns
    data = data[lcols]
    data=cats2oneHot(data,lcols)
    train_pred, test_pred, cur_accuracy = run_linear(RandomForestRegressor(n_jobs=-1), data, y, skf)
    save_level1(train_pred, test_pred, cur_accuracy+'NB_'+dataset+param_type)
    
    
    # #### knn on data1
    dataset, param_type ='data1', '_vc'
    data, y, df, train, test = loadDataset(dataset)
    lcols=data.select_dtypes(include=['object']).columns
    scl = preprocessing.StandardScaler()
    data = data[lcols]
    data = data.fillna(data.mean())
    data = scl.fit_transform(data)
    train_pred, test_pred, cur_accuracy = run_linear(KNeighborsRegressor(n_neighbors=1, n_jobs=-1), pd.DataFrame(data), y, skf)
    save_level1(train_pred, test_pred, cur_accuracy+'knn_'+dataset+param_type)  