"""
   Blend on level2 and create submission

__author__

   Alex Tselikov < atselikov@gmail.com >

"""

from utils_tanz import *
from ats_functions import *
from ml_params import *
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    #load target
    df, _, _, y, skf = load_data(dfold)
    
    #load level1 predictions
    test = np.concatenate((np.loadtxt('level2/blend_test30_1812_mc.txt'),
                           #np.loadtxt('blend_test30_1712_mc_ux2.txt'),
                           #np.loadtxt('blend_test10_1612_mc7.txt'),
                           #np.loadtxt('blend_test10_1612_mc7.txt'),
                          ),axis=-1)
    train = np.concatenate((np.loadtxt('level2/blend_train30_1812_mc.txt'),
                           #np.loadtxt('blend_train30_1712_mc_ux2.txt'),
                           #np.loadtxt('blend_train10_1612_mc7.txt'),
                           #np.loadtxt('blend_train10_1612_mc7.txt'),
                          ),axis=-1)
    print (np.shape(train), np.shape(test))
    
    res_train = np.mean(train, axis=1)#/np.shape(train)[1]
    res_test = np.mean(test, axis=1)#/np.shape(test)[1]
    
    #find best thresholds
    lin2classes_accuracy_threshold(y, res_train, print_ab=True)
    print ('Check optimal thresholds!')
    yres = applyThresholds(res_test.copy(), a, b)

    print ('Check class distribution')
    print (pd.Series(yres).value_counts(normalize=True))
    
    write_submission(yres, 'linKoeff_30xgb')