class DataConfig:
    target_col = 'fraud'
    datetime_col = 'timestamp'
    series_cols = [
        'authorised_times',
        'received_dates',
        'ip_dates',
        'billing_address_dates',
        'delivery_address_dates'
    ]
    list_cols = ['device']


class ModelConfig:
    lgb_params = {
        'objective': 'binary',
        'is_unbalance': True,
        'learning_rate': 0.02,
        'max_depth': 10,
        'num_leaves': 256,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'feature_fraction': 0.33,
        'min_data_in_leaf': 128,
        'nthread': 10,
    }
