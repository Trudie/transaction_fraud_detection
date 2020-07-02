import pandas as pd
import lightgbm as lgb
from sklearn.base import TransformerMixin, BaseEstimator

from config import DataConfig, ModelConfig
from models.utils import train_valid_split
from pre_processing import PreProcessor, drop_refused_by_bank
from feature_engineering import ChecksDecomposor, LagFeaturesExtractor
from auto_encoder import AutoEncoderProjector

SPLIT_DATE = '2016-04-20'


class FraudClassifier(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.embedding_cols = ['authorised_times', 'received_dates', 'ip_dates']
        self.drop_cols = ['first_6_digits', 'payment_ref', 'user_id', 'amount_eur',
                          'billing_address_dates_len', 'delivery_address_dates_len', 'device_len']
        leak_cols = ['refused_by_adyen_risk', 'refused_by_bank', 'is_fraud']
        self.drop_cols += leak_cols + DataConfig.series_cols + DataConfig.list_cols + [DataConfig.datetime_col]
        self.weight_col = 'amount_eur'

        self.pre_processing = PreProcessor()
        self.check_decomposor = ChecksDecomposor()
        self.lag_feature_extractor = LagFeaturesExtractor()
        self.aep = []
        for _ in self.embedding_cols:
            self.aep.append(AutoEncoderProjector())

        self.model = None

    def fit(self, df):

        df = df.sort_values(DataConfig.datetime_col).reset_index(drop=True)
        df = drop_refused_by_bank(df)
        df[DataConfig.target_col] = df['refused_by_adyen_risk'] | df['is_fraud']

        # Split into train and valid set
        train_df, valid_df = train_valid_split(df, split_date=SPLIT_DATE)

        # Preprocessing
        train_df = self.pre_processing.fit_transform(train_df)
        valid_df = self.pre_processing.transform(valid_df)

        # Feature Engineering
        train_df = self.check_decomposor.fit_transform(train_df)

        train_df = self.lag_feature_extractor.fit_transform(train_df)
        for col, aep in zip(self.embedding_cols, self.aep):
            train_embed = aep.fit_transform(train_df[col])
            train_df = pd.concat([train_df, train_embed], axis=1)

        valid_df = self.check_decomposor.transform(valid_df)
        valid_df = self.lag_feature_extractor.transform(valid_df)
        for col, aep in zip(self.embedding_cols, self.aep):
            valid_embed = aep.transform(valid_df[col])
            valid_df = pd.concat([valid_df, valid_embed], axis=1)

        # Split X, y
        train_X = train_df.drop(columns=self.drop_cols + [DataConfig.target_col])
        train_y = train_df[DataConfig.target_col]

        valid_X = valid_df.drop(columns=self.drop_cols + [DataConfig.target_col])
        valid_y = valid_df[DataConfig.target_col]

        # LGB data format
        d_train = lgb.Dataset(train_X, label=train_y)

        d_valid = lgb.Dataset(valid_X, label=valid_y)

        # training
        self.model = lgb.train(
            ModelConfig.lgb_params,
            train_set=d_train,
            valid_sets=[d_train, d_valid],
            num_boost_round=2000,
            early_stopping_rounds=200,
            verbose_eval=200
        )
        return self

    def predict(self, X):
        X = self.pre_processing.transform(X)
        X = self.lag_feature_extractor.transform(X)

        X = self.check_decomposor.transform(X)
        X = self.lag_feature_extractor.transform(X)
        for col, aep in zip(self.embedding_cols, self.aep):
            embed = aep.transform(X[col])
            X = pd.concat([X, embed], axis=1)

        X = X.drop(columns=self.drop_cols + [DataConfig.target_col])
        pred_y = self.model.predict(X)

        return pred_y
