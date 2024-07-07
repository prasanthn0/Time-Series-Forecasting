import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import logging

# Define constants
NUM_FOLD = 5
RANDOM_STATE = 42
N_JOBS = -1
VERBOSE = 1

class OptunaSearch:

    def __init__(self, data, target_column, test_size=0.2, cv_method='kfold', n_trials=100):
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.models = {}
        self.cv_method = cv_method
        self.n_trials = n_trials
        
        # Prepare logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Split the data
        self.train_X, self.test_X, self.train_Y, self.test_Y = self.split_data()
        
        # Initialize and add models automatically
        self.initialize_models()

    def split_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=self.test_size, shuffle=False if self.cv_method == 'timeseries' else True, random_state=RANDOM_STATE)
        self.logger.info("Data split into train and test sets.")
        return train_X, test_X, train_Y, test_Y

    def add_model(self, name, objective_func):
        self.models[name] = objective_func
        self.logger.info(f"Model {name} added with Optuna objective function.")

    def tune_model(self, name):
        if name not in self.models:
            self.logger.error(f"Model {name} not found!")
            return None
        
        objective_func = self.models[name]
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_func, n_trials=self.n_trials)
        trial = study.best_trial
        best_par = trial.params
        
        self.logger.info(f"Model {name} tuned. Best parameters: {best_par}")
        return best_par

    def evaluate_model(self, model):
        predictions = model.predict(self.test_X)
        mse = mean_squared_error(self.test_Y, predictions)
        self.logger.info(f"Model evaluation completed. MSE: {mse}")
        return mse

    def run(self):
        results = {}
        for name in self.models:
            self.logger.info(f"Tuning model: {name}")
            best_params = self.tune_model(name)
            if best_params is not None:
                model = self.get_model_instance(name, best_params)
                model.fit(self.train_X, self.train_Y)
                score = self.evaluate_model(model)
                results[name] = {'model': model, 'score': score}
        return results

    def get_model_instance(self, name, params):
        if name == 'RFR':
            return RandomForestRegressor(**params)
        elif name == 'GBM':
            return GradientBoostingRegressor(**params)
        elif name == 'LGBM':
            return LGBMRegressor(**params)
        elif name == 'XGB':
            return XGBRegressor(**params)
        elif name == 'AdaBoost':
            return AdaBoostRegressor(**params)
        elif name == 'ExTree':
            return ExtraTreesRegressor(**params)
        elif name == 'Bag':
            return BaggingRegressor(**params)
        else:
            raise ValueError(f"Unsupported model name: {name}")

    def initialize_models(self):
        # Define objective functions for each model
        self.add_model('RFR', self.RFR())
        self.add_model('GBM', self.GBM())
        self.add_model('LGBM', self.LGBM())
        self.add_model('XGB', self.XGB())
        self.add_model('AdaBoost', self.AdaBoost())
        self.add_model('ExTree', self.ExTree())
        self.add_model('Bag', self.Bag())

    # Define objective functions for each model
    def RFR(self):
        def objective(trial):
            _n_estimators = trial.suggest_int("n_estimators", 50, 200)
            _max_depth = trial.suggest_int("max_depth", 5, 20)
            _min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            _min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
            _max_features = trial.suggest_int("max_features", 10, 50)

            rf = RandomForestRegressor(
                n_estimators=_n_estimators,
                max_depth=_max_depth,
                min_samples_split=_min_samples_split,
                min_samples_leaf=_min_samples_leaf,
                max_features=_max_features
            )
            return cross_val_score(rf, self.train_X, self.train_Y, n_jobs=-1, cv=2).mean()

        return objective

    def GBM(self):
        def objective(trial):
            loss = trial.suggest_categorical("loss", ['ls', 'lad', 'huber', 'quantile'])
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 1)
            criterion = trial.suggest_categorical("criterion", ['friedman_mse', 'mse', 'mae'])
            max_features = trial.suggest_categorical("max_features", ['auto', 'sqrt', 'log2'])
            gbr = GradientBoostingRegressor(
                loss=loss,
                learning_rate=learning_rate,
                criterion=criterion,
                max_features=max_features
            )
            return cross_val_score(gbr, self.train_X, self.train_Y, n_jobs=-1, cv=3).mean()

        return objective

    def LGBM(self):
        def objective(trial):
            max_depth = trial.suggest_int('max_depth', 1, 20)
            max_bin = trial.suggest_int('max_bin', 10, 500)
            n_estimators = trial.suggest_int('n_estimators', 10, 5000)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.9)
            subsample = trial.suggest_loguniform('subsample', 0.1, 1)
            colsample_bytree = trial.suggest_loguniform('colsample_bytree', 0.1, 1)

            lgbmr = LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                max_depth=max_depth,
                max_bin=max_bin
            )
            return cross_val_score(lgbmr, self.train_X, self.train_Y, n_jobs=-1, cv=2).mean()

        return objective

    def XGB(self):
        def objective(trial):
            max_depth = trial.suggest_int('max_depth', 1, 20)
            n_estimators = trial.suggest_int('n_estimators', 10, 5000)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.9)
            subsample = trial.suggest_loguniform('subsample', 0.1, 1)
            colsample_bytree = trial.suggest_loguniform('colsample_bytree', 0.1, 1)

            xgb = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                max_depth=max_depth
            )
            return cross_val_score(xgb, self.train_X, self.train_Y, n_jobs=-1, cv=2).mean()

        return objective

    def AdaBoost(self):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 5000)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.9)
            loss = trial.suggest_categorical("loss", ['linear', 'square', 'exponential'])

            ada = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss
            )
            return cross_val_score(ada, self.train_X, self.train_Y, n_jobs=-1, cv=2).mean()

        return objective

    def ExTree(self):
        def objective(trial):
            criterion = trial.suggest_categorical("criterion", ['friedman_mse', 'mse', 'mae'])
            max_features = trial.suggest_categorical("max_features", ['auto', 'sqrt', 'log2'])
            n_estimators = trial.suggest_int('n_estimators', 10, 5000)
            max_depth = trial.suggest_int('max_depth', 1, 20)

            ext = ExtraTreesRegressor(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                max_features=max_features
            )
            return cross_val_score(ext, self.train_X, self.train_Y, n_jobs=-1, cv=2).mean()

        return objective

    def Bag(self):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 5000)
            max_samples = trial.suggest_int('max_samples', 1, 100)

            bag = BaggingRegressor(
                n_estimators=n_estimators,
                max_samples=max_samples
            )
            return cross_val_score(bag, self.train_X, self.train_Y, n_jobs=-1, cv=2).mean()

        return objective

if __name__ == "__main__":
    data = pd.read_csv('your_time_series_data.csv')

    # Initialize OptunaSearch with desired scoring method and cross-validation method
    optuna_search = OptunaSearch(data=data, target_column='target_column', cv_method='timeseries')

    # Run the optimization and evaluate models
    results = optuna_search.run()

    # Print results
    for model_name, result in results.items():
        print(f"Model: {model_name}, Score: {result['score']}")

