from datetime import datetime
import pandas as pd
from config import DataConfig


def train_valid_split(df: pd.DataFrame, split_date: str = None, valid_size: float = 0.2):
    """Split training and validation by datetime
    Parameters
    ----------
        df: pd.DataFrame 
            Full dataset with `timestamp` column.
        split_date: string
            Split the data by `split_date`, with format `%Y-%m-%d`
        valid_size: float
            The proportion of the dataset to include in the valid split.
    """
    
    df = df.sort_values(DataConfig.datetime_col)
    
    if split_date is None:
        split_date = df[DataConfig.datetime_col].iloc[-int(len(df) * valid_size)]
        split_date = split_date.replace(minute=0, hour=0, second=0)
    else:
        split_date = datetime.strptime(split_date, '%Y-%m-%d')

    return (
        df.loc[df[DataConfig.datetime_col] < split_date], 
        df.loc[df[DataConfig.datetime_col] >= split_date]
    )
