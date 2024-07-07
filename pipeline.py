

from config import *
from logger import getLogger  



import pandas as pd
import numpy as np
import os
import datetime
import warnings
warnings.filterwarnings('ignore')
import pickle
from statistics import mode
import statsmodels
import statsmodels.api as sm
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from tbats import TBATS, BATS
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from sklearn import metrics
from joblib import Parallel, delayed

from ta import add_all_ta_features

#import boto3,botocore
#import pytz
from datetime import timedelta, datetime
import time
import xlsxwriter

from utility import modelling_utility_func
from utility import feature_selection_func
from utility import feature_extraction_func
from utility import preprocessing_func
from utility import ts_multivariate_func
from utility import ts_univariate_func
from config import *

from sklearn.feature_selection import RFE, RFECV
from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso ,BayesianRidge ,SGDRegressor ,Lars , LassoLars, LassoLarsIC
from sklearn.linear_model import OrthogonalMatchingPursuit,ARDRegression ,MultiTaskElasticNet , MultiTaskLasso, HuberRegressor,RANSACRegressor ,TheilSenRegressor , PoissonRegressor 
from sklearn.linear_model import TweedieRegressor,GammaRegressor,PassiveAggressiveRegressor,enet_path,lars_path,enet_path

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor


class XYZ:


    def main(self):

        # all_feature_df = data_clean.excel_prep.get_data(INPUT_FILE_X,NO_OF_SHEETS,INPUT_FILE_Y)
        all_feature_df = pd.DataFrame()


        train_X = all_feature_df.loc[ : , all_feature_df.columns != RESPONSE_VAR]
        train_Y = all_feature_df[[RESPONSE_VAR]]
        print(train_Y.columns)


    def run_multivariate_parallel(all_feature_df,response_var, lag_start, add_lag,add_delta,
                                add_rolling_mean,cv_threshold,scaler_info ,
                                feature_selection_info , hyperparameter,normalize_error_info,
                                remove_commodity_price,dev_range,freq,
                                test_period, num_fold, model_list,commodity,y_log_scale,
                                num_cores=4):

        obj = ts_multivariate_func.RunMultivariatePipeline(all_feature_df,
                                                        response_var,
                                                        lag_start,
                                                        add_lag,
                                                        add_delta,
                                                        add_rolling_mean,
                                                        cv_threshold,
                                                        scaler_info ,
                                                        feature_selection_info ,
                                                        hyperparameter,
                                                        normalize_error_info,
                                                        remove_commodity_price,
                                                        dev_range,
                                                        freq,
                                                        commodity,y_log_scale)

        res = Parallel(n_jobs=num_cores)(delayed(obj.run_models)(test_period,
                                                    num_fold,
                                                    i, # iterate.........
                                                    model_list) for (i) in list(model_list))
        print(f'Cross Validation Done for Lag: {lag_start}')

        return res



    def tabulate_results(result,lag_start_list,result_type = 'future_forecast',base_val=None):
        """
        result : 
        
        """
        if result_type not in RESULT_TYPES:
            raise Exception('Incorrect Result Type')
        
        final_output = pd.DataFrame()
        for i in range(1,len(lag_start_list)+1): # for each lag
            r = result[i-1]
            idx = modelling_utility_func.future_dates(pd.DatetimeIndex(pd.Series(np.datetime64(TAKE_ACTUAL_TILL))),
                                                    FREQ[0], list(r[0])[0])[-1:]

            if result_type == 'top_features':
                l = r[0][list(r[0])[0]]['top_features'].tolist()
                temp = pd.DataFrame(columns=['Lag'+str(list(r[0])[0])],index=range(len(l)))
                for j in range(len(temp)):
                    temp.loc[j]=l[j]
                final_output = pd.concat([final_output,temp],axis=1)
            else:
                if result_type == 'future_forecast':
                    dummy_df = pd.DataFrame(columns=range(len(r)),index=idx)
                else:
                    dummy_df = pd.DataFrame(columns=range(len(r)),index=['Lag'+str(list(r[0])[0])])

                for m in range(len(r)): # for each model
                    dummy_df.rename(columns={m: r[m][list(r[0])[0]]['model_name']},inplace=True)
                    dummy_df[r[m][list(r[0])[0]]['model_name']] = r[m][list(r[0])[0]][result_type]

                final_output = final_output.append(dummy_df)

        return final_output




    def select_best_model(result_dictionary,cv_metric='mape',avg_selection=False,avg_n_model=5,
                            horizon_bucketing=False,bucket_interval=5):
            """
            result_dict : dictionary, that contains cross validation error & future forecast information
            cv_metric : what metric from Cross Validation to be used for best model selection
            avg_selection : Boolean, if True then forecast is given by averaging some top models else
            forecast is given by the top most model
                Returns the Future Forecast from the best model(s)
            """
            
            
            inp = pd.read_excel(INPUT_FILE_Y)
            inp = inp[(inp['Date'] == TAKE_ACTUAL_TILL)]
            

            # print(inp[RESPONSE_VAR])
            
            result_dict = result_dictionary.copy()
            if cv_metric not in result_dict:
                raise Exception('Invalid Metric')
            ########################################
            forecast_df = result_dict['future_forecast'].copy()
            error_df = result_dict[cv_metric].copy()
            ########################################
            if horizon_bucketing:
                bins = np.arange(0,LAG_START_LIST[-1]+1,bucket_interval)
                labels = [f'Lag{x+1}-Lag{x+bucket_interval}' for x in bins[:-1]]
        #         len(bins),len(labels)
                error_df['bucket']=pd.cut(range(1,len(error_df)+1), bins=bins,labels=labels)
                agg_error_df = error_df.groupby(['bucket']).mean()

                bm = agg_error_df.idxmin(axis=1).tolist() # best model name
                bm = np.repeat(bm,bucket_interval)
                bme = agg_error_df.min(axis=1).tolist() # best model cross validation error
                bme = np.repeat(bme,bucket_interval)
                for icount,i in enumerate(bm): # forecast from the best model
                    forecast_df.loc[forecast_df.iloc[icount].name,
                                    'Forecast']=round(forecast_df.iloc[icount][i],1)
                    forecast_df.loc[forecast_df.iloc[icount].name,'Model']=i
                    forecast_df.loc[forecast_df.iloc[icount].name,'Range']= round((forecast_df.iloc[icount][i]),1)
                    forecast_df.loc[forecast_df.iloc[icount].name,'Error']= bme[icount]
                    forecast_df.loc[forecast_df.iloc[icount].name,
                                    'Direction']='Down' if (round(np.mean(forecast_df.iloc[icount][i]),1))<inp[RESPONSE_VAR].tolist() else 'Up'
            ########################################
            else:
                if avg_selection:
                    temp_df1 = error_df.apply(lambda x: pd.Series(x.nsmallest(avg_n_model).index),axis=1)
                    bm = temp_df1.values.tolist() # best model names
                    temp_df2 = error_df.apply(lambda x: pd.Series(x.nsmallest(avg_n_model).dropna()),axis=1)
                    bme = temp_df2.values.tolist() #best model cross validation error
                    for icount,i in enumerate(bm): # forecast from the best models
                        forecast_df.loc[forecast_df.iloc[icount].name,
                                        'Forecast']=round(np.mean(forecast_df.iloc[icount][i]),1)
                        forecast_df.loc[forecast_df.iloc[icount].name,'Model']='-'.join(i)
                        forecast_df.loc[forecast_df.iloc[icount].name,'Min_Range']= round(np.min(forecast_df.iloc[icount][i]),1)
                        forecast_df.loc[forecast_df.iloc[icount].name,'Max_Range'] = round(np.max(forecast_df.iloc[icount][i]),1)
                        forecast_df['Range'] = forecast_df['Min_Range'].apply(str)+"-"+forecast_df['Max_Range'].apply(str)
                        forecast_df.loc[forecast_df.iloc[icount].name,'Error']= np.nanmean(bme[icount])
                        print("f",(round(np.mean(forecast_df.iloc[icount][i]),1)))
                        print("may",inp[RESPONSE_VAR].tolist())
                        forecast_df.loc[forecast_df.iloc[icount].name,
                                        'Direction']='Down' if (round(np.mean(forecast_df.iloc[icount][i]),1))<inp[RESPONSE_VAR].tolist() else 'Up'
                        
                        inp[RESPONSE_VAR] = round(np.mean(forecast_df.iloc[icount][i]),1)
                        print(inp[RESPONSE_VAR])

                else:

                    bm = error_df.idxmin(axis=1).tolist() # best model name
                    bme = error_df.min(axis=1).tolist() # best model cross validation error
                    for icount,i in enumerate(bm): # forecast from the best model
                        forecast_df.loc[forecast_df.iloc[icount].name,
                                        'Forecast']=round(forecast_df.iloc[icount][i],1)
                        forecast_df.loc[forecast_df.iloc[icount].name,'Model']=i
                        forecast_df.loc[forecast_df.iloc[icount].name,'Range']= round((forecast_df.iloc[icount][i]),1)
                        forecast_df.loc[forecast_df.iloc[icount].name,'Error']= bme[icount]
                        forecast_df.loc[forecast_df.iloc[icount].name,
                                        'Direction']='Down' if ((round(np.mean(forecast_df.iloc[icount][i]),1))<inp[RESPONSE_VAR].tolist()) else 'Up'


            return forecast_df[['Forecast','Model','Range','Error','Direction']]