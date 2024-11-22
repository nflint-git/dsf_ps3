import hashlib
import numpy as np
import pandas as pd

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.


def create_sample_split(df, id_column: str, training_frac=0.8):
    """
    Create sample split based on ID column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        id_column (str): The name of the column containing IDs.
        training_frac (float): Fraction of data to assign to the training set.

    Returns:
        pd.DataFrame: DataFrame with a 'sample' column containing 'train'/'test' split.
    """
    df[id_column] = df[id_column].astype(str)
    # Generate hash-based normalized values
    df['hash_normalized'] = df[id_column].apply(lambda x: (int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16) % (2**32)) / (2**32))
    
    # Assign 'train' or 'test' based on the hash value
    df['sample'] = df['hash_normalized'].apply(lambda x: 'train' if x < training_frac else 'test')
    
    # Drop the intermediate 'hash_normalized' column
    df.drop(columns=['hash_normalized'], inplace=True)
    
    return df
