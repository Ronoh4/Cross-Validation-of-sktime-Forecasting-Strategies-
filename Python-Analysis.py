import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.compose import make_reduction
from sklearn.metrics import mean_absolute_error
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sklearn.neighbors import KNeighborsRegressor

file_path = 'C:/Users/user/Documents/Datasets/Superstore Sales Data.xlsx'
df = pd.read_excel(file_path)
y = pd.Series(df['Sales'])
forecaster_param_grid = {
    'NaiveForecaster': {
        'strategy': ['last', 'mean']
    },
    'ThetaForecaster': {},
    'KNeighborsRegressor': {
        'n_neighbors': [1, 2, 3, 4, 5]
    }
}
cv = TimeSeriesSplit(n_splits=3)
results = {}
for name, params in forecaster_param_grid.items():
    for param_name, param_values in params.items():
        for param_value in param_values:
            if name == 'KNeighborsRegressor':
                forecaster = make_reduction(eval(name)(**{param_name: param_value}), window_length=24)
            else:
                forecaster = eval(name)(**{param_name: param_value})
            forecaster = TransformedTargetForecaster([('forecast', forecaster)])
            scores = []
            for train_index, test_index in cv.split(y):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                forecaster.fit(y_train)
                y_pred = forecaster.predict(fh=range(1, len(test_index) + 1))
                scores.append(mean_absolute_error(y_test, y_pred))
            key = f'{name}({param_name}={param_value})'
            results[key] = np.mean(scores)
for key, value in results.items():
    print(f'{key}: {value:.2f}')
    



