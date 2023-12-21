import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import ast
from sklearn.metrics import  mean_absolute_error
import warnings
import statsmodels.api as sm
import plotly.graph_objects as go
from  genre_label_helpers import *
warnings.filterwarnings("ignore", category=Warning)
    
#get the keys aka unique labels
def get_unique_labels(labels_series):
    keys = set()
    for label_dict in labels_series:
        keys.update(label_dict.keys())
    return list(keys)

def filter_dictionnary(dict, threshold):
    return {key: value for key, value in dict.items() if value > threshold}

def create_labels_df(threshold = 0.5, path_to_file = 'data/DeBERTa-v3/complete_De-BERTa_plots.csv'):
    """Creates a dataframe with the labels and the ratings for each movie togehther with the movie name and relese year"""
    df = pd.read_csv(path_to_file)
    df = df.drop_duplicates(subset=['wiki_movie_id'])
    df = df.set_index('wiki_movie_id')
    #checking that no duplicate indexes are present
    assert len(df.index) == df.index.nunique()
    df = df['labels'].apply(ast.literal_eval)
    #filtering out labels that are below the threshold
    df = df.apply(lambda x: filter_dictionnary(x, threshold))
    assert len(df.index) == df.index.nunique()
    return df

def filter_label_count(label_count, min_nb_to_keep, max_nb_to_keep): 
    df = label_count.copy()
    df = df[df > min_nb_to_keep]
    df = df[df < max_nb_to_keep]
    df = df.sort_values(ascending=False)
    return df

def filter_labels(df,
                     min_nb_to_keep, 
                     max_nb_to_keep):  
    """Filters out labels that are below the min_nb_to_keep and above the max_nb_to_keep
    and returns the filtered dataframe and the list of labels that are kept"""  
    rating_labels = df.copy()
    label_count = rating_labels.apply(lambda x: list(x.keys())).explode().value_counts()
    label_count = filter_label_count(label_count, min_nb_to_keep, max_nb_to_keep)
    label_names = label_count.index.tolist()
    #only leaving labels that are in the filtered zone 
    rating_labels = rating_labels.apply(lambda x: {k: v for k, v in x.items() if k in label_names})
    #dropping movies that have no labels left
    rating_labels = rating_labels[rating_labels.apply(bool)]
    return rating_labels, label_names

def get_label_dummies(rating_labels):    
    #creating dummy variables for labels
    dummy_variables = pd.get_dummies(rating_labels.apply(lambda x: list(x.keys())).explode())
    dummy_variables = dummy_variables.groupby(dummy_variables.index).sum() # Group by movie ID
    return dummy_variables.astype(int)    

def show_label_counts(rating_labels, top_counts_to_show=5):
    count = rating_labels.apply(lambda x: list(x.keys())).explode().value_counts()
    count =count.sort_values(ascending=False)
    count = count.head(top_counts_to_show)

    fig = go.Figure(data=[go.Bar(x=count.values, y=count.index, orientation='h')])
    fig.update_layout(
        title='Label Counts',
        xaxis=dict(title='Count'),
        yaxis=dict(title='Label'))
    fig.show()

def show_label_means(labels_with_rating, nb_to_show=5): 
    mean_ratings = labels_with_rating.explode('labels').groupby('labels')['averageRating'].mean()
    mean_ratings = mean_ratings.sort_values(ascending=False).head(nb_to_show)
    fig = go.Figure(data=[go.Bar(x=mean_ratings.values, y=mean_ratings.index, orientation='h')])
    fig.update_layout(
        title='Label Means',
        xaxis=dict(title='Mean'),
        yaxis=dict(title='Label'),
    )
    # Show the plot
    fig.show()

def show_model(model,top_to_show=5):
    SHOW_TOP_COEF = top_to_show
    model_df = pd.DataFrame(model.summary().tables[1])

    model_df.columns = model_df.iloc[0]
    model_df.columns = ['Genre', 'Coefficient', 'Standard Error', 't-value', 'p-value', '95% CI Lower', '95% CI Upper']
    model_df = model_df[1:].copy()
    for col in model_df.columns:
        model_df[col] = model_df[col].apply(lambda x: x.data)
    for col in model_df.columns[1:]:
        model_df[col] = model_df[col].astype(float)
    coef = model_df['Coefficient'].iloc[0]
    model_df = model_df[1:].copy()
    filtered_model_df = model_df[model_df['p-value'] <= 0.05]

    top = model_df.nlargest(SHOW_TOP_COEF, 'Coefficient')
    bottom = model_df.nsmallest(SHOW_TOP_COEF, 'Coefficient')
    filtered_model_df = pd.concat([top, bottom])

    # Create a Plotly figure with custom error bars
    fig = go.Figure()

    for index, row in filtered_model_df.iterrows():
        genre = row['Genre']
        coef = row['Coefficient'] 
        std_error = row['Standard Error']
        ci_lower = row['95% CI Lower']
        ci_upper = row['95% CI Upper']
        
        # Add a bar trace with custom error bars
        fig.add_trace(go.Bar(
            x=[coef],
            y=[genre],
            orientation='h',
            error_x=dict(
                type='data',
                symmetric=False,
                array=[ci_upper],
                arrayminus=[ci_lower]
            ),
            name=genre
        ))

    # Customize the layout
    fig.update_layout(
        yaxis_title='Label',
        xaxis_title='Coefficient Value',
        title='Coefficients with Confidence Intervals (95%)',
        barmode='group'  # Use 'group' to display multiple bars per genre
)

# Show the plot
    fig.show()

#for filtering out genres with less than threshold movies
def filter_genres(genres_df,threshold, show_plot=False):    
    genre_counts = genres_df.apply(lambda x: list(x.values())).explode().value_counts()
    genre_counts = genre_counts[genre_counts > threshold]
    genre_counts = genre_counts.sort_values(ascending=True)
    top_genres_value_list = genre_counts.index.tolist()
    genres_df = genres_df.apply(lambda x: {k: v for k, v in x.items() if v in top_genres_value_list})
    genres_df = genres_df[genres_df.apply(bool)]

    if show_plot:
        genre_counts.plot(kind='barh')
        plt.xlabel('Counts')
        plt.ylabel('Genres')
        plt.title('Genre Counts')
        plt.show()
    return genres_df

#doing linear regression on the dataframe with the given columns
def linear_regression(X_column_names,y_column_names, df):
    df_train = df.sample(frac=0.85, random_state=1)
    df_test = df.drop(df_train.index)

    X = sm.add_constant(df_train[X_column_names])
    model = sm.OLS(df_train[y_column_names], X).fit()

    X_test = sm.add_constant(df_test[X_column_names])
    preds = model.predict(X_test)

    mae = mean_absolute_error(df_test[y_column_names], preds)
    return model, mae

#getting genrres in one hot encoding
def get_genre_dummies(genres_df):
    dummy_variables = pd.get_dummies(genres_df.apply(lambda x: list(x.values())).explode())
    dummy_variables = dummy_variables.groupby(dummy_variables.index).sum()  # Group by movie ID
    dummy_variables = dummy_variables.astype(int)
    return dummy_variables
#used for quick testing, given full dataframe with needed inxes filterred out and genre information present, 
# get linear regression model and mae

def full_genre_process(metadata_df, threshold=0):
    genres_df = metadata_df.copy()
    genres_df = metadata_df['movie_genres'].apply(lambda x: transform_genres_string(x))
    genres_df = filter_genres(genres_df, threshold)
    dummy_variables = get_genre_dummies(genres_df)
    genres_df = dummy_variables.merge(metadata_df['averageRating'], left_index=True, right_index= True, how='left').copy()
    X_column_names = [col for col in genres_df.columns if col != 'averageRating' ]
    y_column_names = ['averageRating']
    return linear_regression(X_column_names,y_column_names, genres_df)

def transform_genres_string(genres_str):
    try:
        genres_dict = json.loads(genres_str.replace("'", "\""))  # Replace single quotes with double quotes
        return genres_dict
    except json.JSONDecodeError:
        return {}

