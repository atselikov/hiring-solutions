"""
   Data preprocessing

__author__

   Alex Tselikov < atselikov@gmail.com >

"""

#imports
from utils_tanz import *
from ats_functions import *
from ml_params import *
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    #load prepared data
    df, test, train, y, skf = load_data(dfold)
    data = df.drop(['num_private','wpt_name', 'extraction_type_group', 'extraction_type','payment_type', 'water_quality',\
    'scheme_management', 'district_code', 'region','region_code','subvillage', 'ward',\
    'waterpoint_type_group', 'installer','status_group'], 1)

    #correct gps_height&construction_year by medians
    data.loc[data.construction_year<0, 'construction_year']= np.median(data.construction_year[data.construction_year>0])
    data.loc[data.gps_height==0, 'gps_height'] = np.median(data.gps_height[data.gps_height>0])

    #add data lag
    data['date_recorded'] = pd.to_datetime(data['date_recorded'])
    data['temp'] = pd.to_datetime(max(data['date_recorded']))
    data['date_recorded_lag'] = (data['temp'] - data['date_recorded']).dt.days
    data = data.drop(['date_recorded','temp'],1)

    #fill cats by value_counts
    lcols=data.select_dtypes(include=['object']).columns
    data = cats2valueCounts(data, lcols)

    #tfidf settings
    vectorizerLetter = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), min_df=4, max_features = 10000)
    excludeLetter = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
    svdLetter = TruncatedSVD(n_components=50)
    ica = FastICA(n_components=2, random_state=2017)
    vectorizerWord = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3, stop_words='english', max_features = 10000)
    excludeWord = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    svdWord = TruncatedSVD(n_components=100)
    cls = KMeans(n_clusters=15, n_jobs=-1, random_state=2017)
    tfIdfGoodCols=[]

    #add tf-idf features
    ##letters_kmeans_wpt_name,
    ##letters_fisa_scheme_name0,letters_fisa_scheme_name1
    data, tfIdfGoodCols = addTfIdfFeats(df, data, excludeLetter, vectorizerLetter, svdLetter, ica,['scheme_name'],'letters_FICA_',tfIdfGoodCols)
    data, tfIdfGoodCols = addTfIdfFeats(df, data, excludeLetter, vectorizerLetter, svdLetter, cls,['wpt_name'],'letters_kmeans_',tfIdfGoodCols, clust=True)
    #save data 
    data.to_csv('datasets/data_proc.csv', index=False)