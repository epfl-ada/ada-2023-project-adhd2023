import pandas as pd
import ast 

def basic_one_hot_encoded(df, column_name):  
    """
    Converts a column of dictionaries into a DataFrame where each key is an individual column, 
    and the values are the values of the new columns.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the one-hot encoded column.
    column_name (str): The name of the column to be converted.

    Returns:
    pandas.DataFrame: The DataFrame with the one-hot encoded column converted into individual columns.
    """

    df = df.copy()
    df = df.drop_duplicates(subset=['wiki_movie_id'])
    df = df.set_index('wiki_movie_id')
    df = df[column_name]
    df = df.apply(lambda x: ast.literal_eval(x))
    df = pd.DataFrame(df.tolist(), index=df.index)
    return df

def rename_columns_fill_0_1(df):
    """
    Renames the columns of a DataFrame with the first unique non-null value in each column.
    Fills missing values with 0 and replaces non-zero values with 1.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame with renamed columns and filled values.
    """

    df = df.copy()
    for col in df.columns:
        df.rename(columns={col: df[col].dropna().unique()[0]}, inplace=True)
    df = df.fillna(0)
    df[df != 0] = 1
    return df

def filter_columns(df, lower_bound, upper_bound):
    """
    Filters the columns of a DataFrame based on the number of non-zero values.

    Args:
        df (pandas.DataFrame): The DataFrame to filter.
        lower_bound (int): The lower bound for the number of non-zero values.
        upper_bound (int): The upper bound for the number of non-zero values.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """

    df = df.copy()
    columns_to_delete = []
    for column in df.columns:
        if (df[column] != 0).sum() < lower_bound or (df[column] != 0).sum() > upper_bound:
            columns_to_delete.append(column)

    df = df.drop(columns=columns_to_delete).copy()
    return df

def numerize(df):
    """
    Convert the columns of a DataFrame to integer type. 
    (on creation the columns are sometimes object type )

    Args:
        df (pandas.DataFrame): The DataFrame to be numerized.

    Returns:
        pandas.DataFrame: The numerized DataFrame.
    """

    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype(int)
    return df

def delete_empty_rows(df):
    """
    Delete rows from a DataFrame that contain only zeros.

    Args:
        df (pandas.DataFrame): The DataFrame to remove empty rows from.

    Returns:
        pandas.DataFrame: The DataFrame with empty rows removed.
    """

    df = df.copy()
    df = df.loc[(df != 0).any(axis=1)]
    return df


def get_one_hot_encoded_df(df, column_name, lower_bound, upper_bound, delete_rows=False):
    """
    Performs a series of transformations on a DataFrame to obtain a one-hot encoded DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column_name (str): The name of the column to be one-hot encoded.
        lower_bound (int): The lower bound for the number of non-zero values in the filtered columns.
        upper_bound (int): The upper bound for the number of non-zero values in the filtered columns.
        delete_rows (bool, optional): Whether to delete rows containing only zeros. Defaults to False.

    Returns:
        pandas.DataFrame: The one-hot encoded DataFrame.
    """
    
    df = df.copy()
    df = basic_one_hot_encoded(df, column_name)
    df = rename_columns_fill_0_1(df)
    df = filter_columns(df, lower_bound, upper_bound)
    df = numerize(df)
    if delete_rows:
        df = delete_empty_rows(df)
    return df