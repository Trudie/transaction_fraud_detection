from typing import List
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from datetime import datetime

from config import DataConfig

OUTLIER_BOUND = [0, 250]
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def drop_refused_by_bank(X: pd.DataFrame) -> pd.DataFrame:
    return X.loc[X['refused_by_bank'] == 0].reset_index(drop=True)


class PreProcessor(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.cat_cols = ['merchant',
                         'card_network',
                         'card_type',
                         'bank_country_id',
                         'user_country_id']
        self.list_cols = DataConfig.list_cols + DataConfig.series_cols
        self.cat_map = dict()
        self.ro = RemoveOutlier(lower_bound=OUTLIER_BOUND[0], upper_bound=OUTLIER_BOUND[1])

    def fit(self, X: pd.DataFrame):
        # extract device
        X['device_minor'] = np.nan
        X['device_minor'] = X.loc[X['device'].notnull(), 'device'].apply(lambda x: x[0])
        self.cat_cols.append('device_minor')

        for col in self.cat_cols:
            self.cat_map[col] = X[col].astype('category').cat.categories.values

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # device
        X['device_minor'] = np.nan
        X['device_minor'] = X.loc[X['device'].notnull(), 'device'].apply(lambda x: x[0])
        # extract is same country
        X['is_same_country'] = X['bank_country_id'] == X['user_country_id']

        # X = self.__drop_refused_by_bank(X)
        X = self.__create_target(X)
        X = self.__clean_dtypes(X, self.cat_map)
        X = self.__get_stats_from_list(X, 'risk_checks')
        X = self.__count_len(X, self.list_cols)
        X = self.___extract_time_to_last_auth(X)

        # DoW
        X['dow'] = X[DataConfig.datetime_col].dt.dayofweek

        # truncate outlier
        X['amount_eur_trunc'] = self.ro.truncate(X['amount_eur'])

        return X

    @staticmethod
    def __create_target(X: pd.DataFrame) -> pd.DataFrame:
        X[DataConfig.target_col] = X['refused_by_adyen_risk'] | X['is_fraud']

        return X

    @staticmethod
    def __clean_dtypes(X: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        for col in mapping.keys():
            X[col] = pd.Categorical(X[col], categories=mapping[col], ordered=False)

        return X

    @staticmethod
    def __get_stats_from_list(X: pd.DataFrame, col: str) -> pd.DataFrame:
        """Extract mean and standard deviation from nested list columns
        Parameters
        ----------
        X: pd.DataFrame
            Full dataset, must have `col` in it
        col: string
            The column name of nested list features
        """

        null_idx = X[col].isnull()

        # Score average
        col_name = '_'.join([col, 'avg'])
        X[col_name] = np.nan
        X.loc[~null_idx, col_name] = X.loc[~null_idx, col].apply(lambda x: np.mean([l[0] for l in x]))

        # Score std
        col_name = '_'.join([col, 'std'])
        X[col_name] = np.nan
        X.loc[~null_idx, col_name] = X.loc[~null_idx, col].apply(lambda x: np.std([l[0] for l in x]))

        return X

    @staticmethod
    def __count_len(X: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
        for col in col_names:
            new_name = '_'.join([col, 'len'])
            valid_idx = X[col].notnull()

            X[new_name] = 0
            X.loc[valid_idx, new_name] = X.loc[valid_idx, col].apply(len)

        return X

    @staticmethod
    def ___extract_time_to_last_auth(X):
        def get_last_auth(x):
            if len(x) == 0:
                return np.nan
            else:
                x = datetime.fromtimestamp(x[-1])

            return x.strftime(DATETIME_FORMAT)

        X['last_auth'] = pd.to_datetime(
            X['authorised_times'].apply(get_last_auth),
            format=DATETIME_FORMAT
        )

        X[DataConfig.datetime_col] = pd.to_datetime(
            X[DataConfig.datetime_col], format=DATETIME_FORMAT
        )
        X['time_to_last_auth'] = (X[DataConfig.datetime_col] - X['last_auth']).apply(lambda x: x.days)

        X = X.drop(columns=['last_auth'])

        return X


class RemoveOutlier(TransformerMixin, BaseEstimator):
    def __init__(self, lower_bound=None, upper_bound=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def fit(self, x: pd.Series):
        q1, q3 = np.percentile(x, q=[25, 75])
        bound = (q3 - q1) * 1.5
        self.lower_bound = q1 - bound
        self.upper_bound = q3 + bound

        return self

    def filter(self, x: pd.Series) -> pd.Series:
        mask = (x >= self.lower_bound) & (x <= self.upper_bound)
        x = x.loc[mask]

        return x

    def truncate(self, x: pd.Series) -> pd.Series:
        x.loc[x < self.lower_bound] = self.lower_bound
        x.loc[x > self.upper_bound] = self.upper_bound

        return x
