import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor, ARDRegression, HuberRegressor
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import logging

# Define constants
NUM_FOLD = 5
RANDOM_STATE = 42
N_JOBS = -1
VERBOSE = 1

class GridSearch:
    
    def __init__(self, data, target_column, test_size=0.2, scoring='neg_mean_squared_error', cv_method='kfold'):
        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.models = {}
        self.cv_method = cv_method
        self.scoring = scoring
        
        # Prepare logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Determine cross-validation method
        if cv_method == 'kfold':
            self.cv = KFold(n_splits=NUM_FOLD, shuffle=True, random_state=RANDOM_STATE)
        elif cv_method == 'timeseries':
            self.cv = TimeSeriesSplit(n_splits=NUM_FOLD)
        else:
            raise ValueError("Invalid cv_method. Choose 'kfold' or 'timeseries'.")
        
        # Split the data
        self.train_X, self.test_X, self.train_Y, self.test_Y = self.split_data()
        
        # Initialize and add models
        self.initialize_models()

    def split_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=self.test_size, shuffle=False if self.cv_method == 'timeseries' else True, random_state=RANDOM_STATE)
        self.logger.info("Data split into train and test sets.")
        return train_X, test_X, train_Y, test_Y
    
    def add_model(self, name, model_class, param_grid):
        self.models[name] = {'model': model_class, 'param_grid': param_grid}
        self.logger.info(f"Model {name} added with parameter grid: {param_grid}")

    def tune_model(self, name):
        if name not in self.models:
            self.logger.error(f"Model {name} not found!")
            return None
        
        model_info = self.models[name]
        model = model_info['model']
        param_grid = model_info['param_grid']
        
        gsearch = GridSearchCV(estimator=model(),
                               param_grid=param_grid,
                               cv=self.cv,
                               n_jobs=N_JOBS,
                               scoring=self.scoring,
                               verbose=VERBOSE)

        gsearch.fit(self.train_X, self.train_Y)
        self.logger.info(f"Model {name} tuned. Best parameters: {gsearch.best_params_}")
        return gsearch.best_estimator_

    def evaluate_model(self, model):
        predictions = model.predict(self.test_X)
        if self.scoring == 'neg_mean_squared_error':
            score = mean_squared_error(self.test_Y, predictions)
        elif self.scoring == 'neg_mean_absolute_error':
            score = mean_absolute_error(self.test_Y, predictions)
        elif self.scoring == 'r2':
            score = r2_score(self.test_Y, predictions)
        else:
            raise ValueError(f"Unsupported scoring method: {self.scoring}")
        self.logger.info(f"Model evaluation completed. Score: {score}")
        return score

    def run(self):
        results = {}
        for name in self.models:
            self.logger.info(f"Tuning model: {name}")
            best_model = self.tune_model(name)
            if best_model is not None:
                score = self.evaluate_model(best_model)
                results[name] = {'model': best_model, 'score': score}
        return results

    def initialize_models(self):
        # Define the models and their parameter grids
        model_params = [
            ('Linear', LinearRegression, {'n_jobs': [1]}),
            ('Ridge', Ridge, {
                'solver': ['svd', 'cholesky', 'lsqr', 'sag'],
                'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }),
            ('Lasso', Lasso, {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}),
            ('ElasticNet', ElasticNet, {
                "max_iter": [1, 5, 10],
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "l1_ratio": np.arange(0.0, 1.0, 0.1)
            }),
            ('BayesianRidge', BayesianRidge, {
                'alpha_init': [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9],
                'lambda_init': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9]
            }),
            ('SGDRegressor', SGDRegressor, {
                'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'max_iter': [100, 1000, 10000],
                'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                'penalty': ['l2', 'l1', 'elasticnet']
            }),
            ('ARDRegression', ARDRegression, {
                'alpha_1': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'n_iter': [100, 1000, 10000]
            }),
            ('HuberRegressor', HuberRegressor, {
                'epsilon': [1, 1.5, 2, 2.5, 3],
                'max_iter': [100, 1000, 10000],
                'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
            })
        ]

        # Add all models to the grid search
        for name, model, param_grid in model_params:
            self.add_model(name, model, param_grid)


if __name__ == "__main__":
    data = pd.read_csv('your_time_series_data.csv')

    # Initialize GridSearch with desired scoring method and cross-validation method
    gs = GridSearch(data=data, target_column='target_column', scoring='neg_mean_absolute_error', cv_method='timeseries')

    # Run the grid search and evaluate models
    results = gs.run()

    # Print results
    for model_name, result in results.items():
        print(f"Model: {model_name}, Score: {result['score']}")
