"""
   Create metafeatures for level2

__author__

   Alex Tselikov < atselikov@gmail.com >

"""

from utils_tanz import *
from ats_functions import *
from ml_params import *
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":

	#load prepared data
	df, test, train, y, skf = load_data(dfold)
	data = pd.read_csv('datasets/data_proc.csv')
	
	#run bagging stage
	bags=30
	blend_train = np.zeros((len(y), bags))
	blend_test = np.zeros((len(data[len(y):]), bags))
	t1=time()
	for j in range(bags): 
	    train_pred, test_pred, cur_accuracy = run_mc_xgb(xgb_params_opt_mc_hyper_eta, data, y, skf, 100, seed=300+j, CalcTest=True)
	    blend_train[:, j] = train_pred
	    blend_test[:, j] = test_pred
	    print ('-'*30)
	    print (j, cur_accuracy, np.round((time() - t1) / 60. ,3))
	np.savetxt('level2/blend_train30_1812_mc.txt',blend_test)
	np.savetxt('level2/blend_train30_1812_mc.txt',blend_train)  	