import logging
import os
"""
All the variables are defined in Upper Case
"""

COMMODITY = 'Shipping container'

#-------------------------- PROJECT INFO ------------------------------------------ #

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_PATH  = os.path.join(PROJECT_PATH, "outputs") 
INPUT_PATH = os.path.join(PROJECT_PATH, "inputs")

#-------------------------- Input and Output Files -------------------------------- #

# INPUT_FILE_X = "/home/ubuntu/ananya_container_pf/scpf_codes_sm/Input/data_for_freight_prediction.xlsx"
# INPUT_FILE_X = PROJECT_PATH+"Input/data_for_freight_prediction.xlsx"     #class parameter
# NO_OF_SHEETS = 5                                                                                     
# INPUT_FILE_Y = PROJECT_PATH+"Input/sea_freight_data.xlsx"              
# OUTPUT_PATH  = PROJECT_PATH+"Outputs/Modelling/Monthly/"               
# FINAL_OUTPUT = PROJECT_PATH+"Forecast_Output/Forecast_Results.xlsx"             
# CONFIG_RES   = PROJECT_PATH+"Forecast_Output/Config_Res.xlsx"                                             
ERROR_TYPE   = ['mape','mae']                                                                         

#-------------------------- Date ranges ------------------------------------------ #

ANALYSIS_START_DATE = '2020-01-01'               # *data should atleast start from this date
ANALYSIS_END_DATE = None
HIST_DATE = '2022-12-01'                         # till this date historical data is downloaded
TAKE_ACTUAL_TILL = '2022-06-01'

#-------------------------- Model Variables and Parameter ------------------------- #
RESPONSE_VAR = "china"
ALL_RESPONSE_VAR = ['China','USA','Japan','South Korea','Europe main Ports','Brazil','Turkey']
# ALL_RESPONSE_VAR = ['China']
FREQ = 'Monthly'                                  # forecast time level (Daily/Weekly/Monthly)
VAR_REQ_FREQ = 'Monthly'                          # upto which level of tickers to be used
FORECAST_PERIOD = [1,2,3]                          # month ahead forecasts
BACKTEST_PERIOD = 6
MULTIVARIATE_MODELS = ['Ridge','SGDR','BR','ElasticNet','Lasso','ARDR','HR','RFR','GBM',
             'Adaboost','Ext','XGB','LGBM'] 
UNIVARIATE_MODELS = ['AR','AUTOARIMA','ETS','Holt']
RUN_UNIVARIATE = True
OPTUNA_TRIALS = 10
EARLY_STOPPING = False                            # Set to True to activate Early Stopping in optuna
OPTUNA_EARLY_STOPING = 15                         # Hyper parameter tuning

#-------------------------- Flags for average forecasts ---------------------------- #
AVG_PREDS =  False
ONLY_Y_ROLL = False                                # True / False

# based on freq & avg pred 
if AVG_PREDS:
    if ONLY_Y_ROLL:
        VERSION = 'V1'
    else:
        VERSION = 'V2'
if AVG_PREDS:
    if FREQ == 'Weekly':
        ROLL_WINDOW = 4
        LAG_START_LIST = np.arange(4,49,4)
        TEST_PERIOD = 24
    if FREQ == 'Daily':
        ROLL_WINDOW = 30
        LAG_START_LIST = np.arange(30,181,30)
        TEST_PERIOD = 30
    if FREQ == 'Monthly':
        raise Exception()
else:
    if FREQ == 'Weekly':
        LAG_START_LIST = np.arange(1,49,1)
        TEST_PERIOD = 24 # last n weeks for testing in cross validation
    if FREQ == 'Daily':
        LAG_START_LIST = np.arange(1,61,1)
        TEST_PERIOD = 90 # last n days for testing in cross validation
    if FREQ == 'Monthly':
#         LAG_START_LIST = np.arange(1,FORECAST_PERIOD+1,1)  # forecast period 
#         LAG_START_LIST = [6] # test
        TEST_PERIOD = 3 # last n months for testing in cross validation
        

RUN_TECHNICAL_INDICATOR_FEATURE = False           # Features from Techinal Indicators 
RUN_TSFRESH_FEATURE = False                       # Features from TS Fresh Package
REMOVE_EQUITY = False                             # True means Equity Related Tickers will be removed
REMOVE_COMMODITY_PRICE = False                    # True means COMMODITY Price related Tickers will be removed
ADD_LAG = 6                                       # i.e. -  np.arange(lag_start, lag_start+add_lag, 1)
ADD_DELTA = False                                 # True/False,
ADD_ROLLING_MEAN = True                           # True/False,
CV_THRESHOLD = 0.1                                # coefficient of variation threshold

ONLY_IMP_TICKER = False
if ONLY_IMP_TICKER:
    IMP_VERSION = 'V2'
    REMOVE_COMMODITY_PRICE = True
REMOVE_EQUITY = False                             # True means Equity Related Tickers will be removed
PERC_CHANGE = False       
          
FIELDS=  None                                    # which fields to select , put None to select OHLCV
ALL_FIELDS = ['PX_OPEN','PX_HIGH','PX_LOW','PX_LAST','PX_VOLUME'] 
if RUN_TECHNICAL_INDICATOR_FEATURE:
    FIELDS = ALL_FIELDS.copy()

#-------------------GLOBAL VARIABLES: MODELLING  ----------------#

N_JOBS = -1
VERBOSE = 1
PRE_COVID_MODEL =  False
RANDOM_STATE = 42                                # Random state for ML algorithms
NUM_FOLD = 3                                     # cross validatin - Walk forward folds

FEATURE_SELECTION_INFO =(True,'RFE',30,0.05)     # True/False, type of feature selection, num of features, RFE_step  RFE RF 
SCALER_INFO = (True,'ss')
HYPERPARAMETER = False 
NORMALIZE_ERROR_INFO = (False,'range') 
DEV_RANGE = 5                                   # range within which the forecast should vary (business input)
NUM_CORES = 4
Y_LOG_SCALE = False
ADD_TIME_COMPONENT = False