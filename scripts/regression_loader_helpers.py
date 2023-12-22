import pandas as pd
import ast 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json

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
    return df

def filter_columns(df, lower_bound , upper_bound ):
    """
    Filters the columns of a DataFrame based on the number of non-zero values.

    Args:
        df (pandas.DataFrame): The DataFrame to filter.
        lower_bound (int): The lower bound for the number of non-zero values.
        upper_bound (int): The upper bound for the number of non-zero values.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """

    if not upper_bound:
        upper_bound = df.shape[0]

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


def get_one_hot_encoded_df(df, column_name, lower_bound = 0, upper_bound = None, delete_rows=False, rename_columns=False):
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
    if not upper_bound:
        upper_bound = df.shape[0]
    if column_name == 'movie_genres':
        gs = get_genres_series(df,lower_bound,upper_bound)
        return get_genre_dummies(gs)
    
    df = basic_one_hot_encoded(df, column_name)
    if rename_columns:
        df = rename_columns_fill_0_1(df)

    df = df.fillna(0)
    df[df != 0] = 1

    df = filter_columns(df, lower_bound, upper_bound)
    df = numerize(df)
    if delete_rows:
        df = delete_empty_rows(df)
    return df

def get_metadata_subset(full_metadata_df, genre = None, period = None, language = None, country = None):
    """
    Choose a subset of the full metadata DataFrame based on genre, language, period, and country.

    Args:
        full_metadata_df (pandas.DataFrame): The full metadata DataFrame.
        genre (str, optional): The genre to filter by. Defaults to None.
        language (str, optional): The language to filter by. Defaults to None.
        period (str, optional): The period to filter by. Defaults to None.
        country (str, optional): The country to filter by. Defaults to None.

    Returns:
        pandas.DataFrame: The subsetted DataFrame.
    """
       
    df = full_metadata_df.copy()
    if genre:
        genres_one_hot = get_one_hot_encoded_df(full_metadata_df, column_name="movie_genres", lower_bound=0, upper_bound = full_metadata_df.shape[0])
        action_ids = genres_one_hot[genres_one_hot[genre] == 1].index.tolist()
        df = df.loc[action_ids]
    if language:
        df = df[df['movie_languages'].apply(lambda x: language in ast.literal_eval(x).values())]
    if period:
        df = df[df['movie_release_year'].apply(lambda x: period[0] < x < period[1])]
    if country:
        df = df[df['movie_countries'].apply(lambda x: country in ast.literal_eval(x).values())]
    return df

def transform_genres_string(genres_str):
    """
    Transforms a string representation of genres into a dictionary.

    Args:
        genres_str (str): The string representation of genres.

    Returns:
        dict: A dictionary representing the genres.

    Raises:
        None
    """

    try:
        genres_dict = json.loads(genres_str.replace("'", "\""))  # Replace single quotes with double quotes
        return genres_dict
    except json.JSONDecodeError:
        return {}
    
def get_genres_series(df, lower_bound, upper_bound):
    """
    Returns a filtered series of movie genres based on the lower and upper bounds.

    Parameters:
    df (DataFrame): The input DataFrame containing the movie genres.
    lower_bound (int): The lower bound for filtering the genres.
    upper_bound (int): The upper bound for filtering the genres.

    Returns:
    Series: A filtered series of movie genres.
    """

    genres_df = df.copy()
    genres_df['movie_genres'] = genres_df['movie_genres'].apply(transform_genres_string)
    genres_df = genres_df['movie_genres']
    genres_df = filter_genres(genres_df, lower_bound, upper_bound)
    return genres_df


def filter_genres(genres_df, lower_bound, upper_bound):
    """
    Filters the genres in the given DataFrame based on their occurrence count.

    Args:
        genres_df (pandas.DataFrame): DataFrame containing genres as columns and their occurrence count as values.
        lower_bound (int): The minimum occurrence count for a genre to be included in the filtered DataFrame.
        upper_bound (int): The maximum occurrence count for a genre to be included in the filtered DataFrame.

    Returns:
        pandas.DataFrame: Filtered DataFrame containing genres that meet the occurrence count criteria.
    """

    genre_counts = genres_df.apply(lambda x: list(x.values())).explode().value_counts()
    genre_counts = genre_counts[genre_counts > lower_bound].astype(int)
    genre_counts = genre_counts.sort_values(ascending=True)
    top_genres_value_list = genre_counts.index.tolist()
    genres_df = genres_df.apply(lambda x: {k: v for k, v in x.items() if v in top_genres_value_list})
    genres_df = genres_df[genres_df.apply(bool)]
    return genres_df.copy()


def get_genre_dummies(genres_df):
    """
    Converts a DataFrame of genres into dummy variables.

    Parameters:
    genres_df (DataFrame): A DataFrame containing genres information.

    Returns:
    DataFrame: A DataFrame with dummy variables representing the genres.
    """
    
    dummy_variables = pd.get_dummies(genres_df.apply(lambda x: list(x.values())).explode())
    dummy_variables = dummy_variables.groupby(dummy_variables.index).sum()  # Group by movie ID
    dummy_variables = dummy_variables.astype(int)
    return dummy_variables
