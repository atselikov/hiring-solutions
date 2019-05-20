
"""
   Params for boosting algos

__author__

    Alex Tselikov < atselikov@gmail.com >

"""

############################################### cat boost

CatBoostDefault = {
    'loss_function':'MultiClass', 
    'eval_metric':'Accuracy', 
    'iterations':5000,
    'od_type':'Iter', 
    #'od_wait':ESR, 
    'use_best_model':True, 
    #'random_seed':2017,
    'classes_count':3, 
    'depth':6, 
    'l2_leaf_reg':3, 
    'learning_rate':0.03, 
    'max_ctr_complexity':4, 
    'rsm':0.9
 } 
 
CatBoostOptimized = {
    'loss_function':'MultiClass', 
    'eval_metric':'Accuracy', 
    'iterations':5000,
    'od_type':'Iter', 
    #'od_wait':ESR, 
    'use_best_model':True, 
    #'random_seed':2017,
    'classes_count':3, 
    'depth':8, 
    'l2_leaf_reg':1.0, 
    'learning_rate':0.09, 
    'max_ctr_complexity':5, 
    'rsm':0.75
 }
 
CatBoostPar1 = {
    'loss_function':'MultiClass', 
    'eval_metric':'Accuracy', 
    'iterations':5000,
    'od_type':'Iter', 
    #'od_wait':ESR, 
    'use_best_model':True, 
    #'random_seed':2017,
    'classes_count':3, 
    'depth':10, 
    'l2_leaf_reg':2.0, 
    'learning_rate':0.01, 
    'max_ctr_complexity':6, 
    'rsm':0.7
 }


############################################### xgb_params

xgb_params_mc0 = {
    "objective": "multi:softmax",
    "learning_rate": 0.01,
    "max_depth": 10,
    #"min_child_weight": 3,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    'num_class': 3
}
xgb_params_mc1 = {
    "objective": "multi:softmax",
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    'num_class': 3
}

xgb_params_mc2 = {
    "objective": "multi:softmax",
    "learning_rate": 0.01,
    "max_depth": 10,
    "min_child_weight": 3,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    'num_class': 3
}

xgb_params_mc3 = {
    "objective": "multi:softmax",
    "learning_rate": 0.1,
    "max_depth": 15,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    'num_class': 3
}

xgb_params_mc4 = {
    "objective": "multi:softmax",
    "learning_rate": 0.05,
    "max_depth": 12,
    "min_child_weight": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    'num_class': 3
}

xgb_params_lin1 = {
    "objective": "reg:linear",
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
}

xgb_params_lin2 = {
    "objective": "reg:linear",
    "learning_rate": 0.01,
    "max_depth": 10,
    "min_child_weight": 3,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
}

xgb_params_lin3 = {
    "objective": "reg:linear",
    "learning_rate": 0.1,
    "max_depth": 15,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

xgb_params_lin4 = {
    "objective": "reg:linear",
    "learning_rate": 0.05,
    "max_depth": 12,
    "min_child_weight": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
}

xgb_params_opt_mc = {
    "objective": "multi:softmax",
    'num_class': 3,
    "learning_rate": 0.02,
    "max_depth": 14,
    "colsample_bytree": 0.4
}

xgb_params_opt_mc_hyper = {
    "objective": "multi:softmax",
    "eval_metric": "merror",
    'num_class': 3,
    "learning_rate": 0.2,
    "max_depth": 13,
    "colsample_bytree": 0.4
}

xgb_params_logd = {
    "objective": "reg:linear",
    "learning_rate": 0.05,
    "max_depth": 10,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

xgb_params_opt = {
    "objective": "reg:linear",
    "learning_rate": 0.2,
    "max_depth": 14,
    "colsample_bytree": 0.4
}

xgb_params_mc = {
    "objective": "multi:softmax",
    "learning_rate": 0.05,
    "max_depth": 10,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    'num_class': 3
}

xgb_params_lin_opt = {
    "objective": "reg:linear",
    "learning_rate": 0.14,
    "gamma": 0.192,
    "max_depth": 8,
    "max_delta_step": 0.08,
    "min_child_weight": 6,
    "subsample": 0.7,
    "colsample_bytree": 0.72
}

xgb_params_opt_mc_hyper_eta = {
    "objective": "multi:softmax",
    "eval_metric": "merror",
    'num_class': 3,
    "learning_rate": 0.02,
    "max_depth": 13,
    "colsample_bytree": 0.4
}

xgb_params_opt_mc_hyper_eta_lin = {
    "objective": "reg:linear",
    #"eval_metric": "merror",
    #'num_class': 3,
    "learning_rate": 0.02,
    "max_depth": 13,
    "colsample_bytree": .4,
    'silent': 1
}

xgb_params_opt_mc_hyper_eta_rank = {
    "objective": "rank:pairwise",
    #"eval_metric": "merror",
    #'num_class': 3,
    "learning_rate": 0.02,
    "max_depth": 13,
    "colsample_bytree": .4,
    'silent': 1
}

xgb_params_opt_mc_hyper_eta_poi = {
    "objective": "count:poisson",
    #"eval_metric": "merror",
    #'num_class': 3,
    "learning_rate": 0.02,
    "max_depth": 13,
    "colsample_bytree": .4,
    'silent': 1
}