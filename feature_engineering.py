import numpy as np
import pandas as pd
from typing import List, Tuple
from collections import Counter
from sklearn.base import TransformerMixin, BaseEstimator
from collections import defaultdict

from config import DataConfig


class ChecksDecomposor(TransformerMixin, BaseEstimator):
    def __init__(self, top_k: int = 10):
        """Decompose `risk_checks` to dummies variables, but only keep `top_k` frequent 
        risk checks' id
        
        Parameters
        ----------
        top_k: int
            Number of check ids wanna keep
        """
        self.top_k = top_k
        self.col_name = 'risk_checks'

    def fit(self, X: pd.DataFrame):
        # Get `top_k` frequent risk checks' id
        all_checks = []

        for _, checks in X[self.col_name].iteritems():
            if checks is not None:
                all_checks += [c for s, c in checks]

        c = Counter(all_checks)
        self.top_checks = [check_id for check_id, _ in c.most_common(self.top_k)]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_df = X[self.col_name].apply(lambda x: self.get_top_dummies(x, self.top_checks)).tolist()
        check_df = pd.DataFrame(data=check_df, columns=self.top_checks, index=X.index)
        check_df = check_df.add_prefix(self.col_name)

        X = pd.concat([X, check_df], axis=1)

        return X.drop(self.col_name, axis=1)

    @staticmethod
    def get_top_dummies(checks: List[Tuple[int, str]], keep_list: List[str]) -> List[int]:
        res = [np.nan] * len(keep_list)

        if checks is None:
            return res

        for s, c in checks:
            try:
                res[keep_list.index(c)] = s
            except:
                pass

        return res


class LagFeaturesExtractor(TransformerMixin, BaseEstimator):
    """ Join user past behavior pattern.

    Extract pattern from earlier transaction, eg. last week spend, and join to current transaction as features.

    Attributes:
        user_transc :
            Hash map for user historical transaction.
            Structure is {user_id: [(transaction_id, transaction_timestamp), ...] }
        agg_func :
            Aggregation function for past.
        time_lag :
            The the window size of earlier transaction data
    """

    def __init__(self):
        self.user_transc = defaultdict(list)
        self.agg_func = ['mean', 'count']
        self.time_lag = ['1 D', '1 W', '30 D']

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame, col_name: str = 'amount_eur', col_groupby='user_id') -> pd.DataFrame:
        """
        :param X: Transaction data with timestamp
        :param col_name: column to be aggregated
        :param col_groupby: column to be grouped by
        :return: data with new features
        """
        X = X.sort_values(by=DataConfig.datetime_col)

        # Initiate new features with np.nan
        new_cols = ['_'.join([col_name, lag, f]) for f in self.agg_func for lag in self.time_lag]
        for col in new_cols:
            X[col] = np.nan
        # each transaction
        for idx, row in X.iterrows():
            user_id, end_time = row[col_groupby], row[DataConfig.datetime_col]
            transc = self.user_transc[user_id]

            if len(transc) > 0:
                for lag in self.time_lag:
                    start_time = end_time - pd.Timedelta(lag)
                    # join stats
                    recent_euro = [e for e, t in transc if t > start_time]
                    recent_euro = pd.DataFrame(recent_euro, columns=[col_name])
                    stats = recent_euro.agg(self.agg_func)

                    for func, stat in stats.iterrows():
                        X.loc[idx, '_'.join([col_name, lag, func])] = stat[col_name]
            # insert user latest transaction
            self.user_transc[user_id].append((row[col_name], end_time))

        return X





