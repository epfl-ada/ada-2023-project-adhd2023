import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import torch


# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-Large-cased')
model = BertModel.from_pretrained('bert-large-cased')

# Function to tokenize and encode text
def encode(text, max_length=512):
    """
    Tokenizes and encodes text using a BERT model.

    Parameters:
    - text (str): The input text to be encoded.
    - max_length (int): The maximum length of the encoded sequence (default is 512).

    Returns:
    - torch.Tensor: The encoded embeddings of the input text.
    """

    # Subtract 2 for [CLS] and [SEP] tokens
    if len(text) == 0:
        print("Empty text")  # Debugging

    max_length -= 2
    tokens = tokenizer.tokenize(text)
    if len(tokens) == 0:
        print("Empty tokens")  # Debugging

    # Split the tokens into chunks of specified max length
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    if not chunks:  # Check if chunks are empty
        print(f"No chunks for text: {text}")  # Debugging

    # Process each chunk
    chunk_embeddings = []
    for chunk in chunks:
        # Add special tokens
        chunk = ['[CLS]'] + chunk + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(chunk)
        input_tensor = torch.tensor([input_ids]).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            # Get the embeddings from the model
            last_hidden_states = model(input_tensor)[0]

        # Append the mean-pooled embeddings from each chunk
        chunk_embeddings.append(last_hidden_states[0].mean(dim=0))

    # Aggregate the embeddings from each chunk (mean pooling here)
    embeddings = torch.mean(torch.stack(chunk_embeddings), dim=0)
    return embeddings


# Read datasets
events = pd.read_csv('data/events.csv')
#convert the year column to int
events['Year'] = events['Year'].astype(int)


#read tsv file and add headers
movie_metadata_df = pd.read_csv('data/movie.metadata.tsv', sep='\t', header=None, 
                names=['wiki_movie_id', 
                        'freebase_movie_id', 
                        'movie_name', 
                        'movie_release_date', 
                        'movie_box_office_revenue', 
                        'movie_runtime', 
                        'movie_languages', 
                        'movie_countries', 
                        'movie_genres'])

#changing the values of outliers
movie_metadata_df.loc[movie_metadata_df['movie_name'] == 'Zero Tolerance', 'movie_runtime'] = 88
movie_metadata_df.loc[movie_metadata_df['movie_name'] == 'Hunting Season', 'movie_release_date'] = '2010-12-02'

#add realase_year 
movie_metadata_df['startYear']= movie_metadata_df['movie_release_date'].str[:4]

#change movie_release_date to pandas datetime
movie_metadata_df['movie_release_date'] = pd.to_datetime(movie_metadata_df['movie_release_date'], format='%Y-%m-%d', errors='coerce')

#load IMDB reviews
rating_id_df = pd.read_csv('data/rating_id.tsv',  sep='\t')
name_id_df = pd.read_csv('data/name_id.tsv',  sep='\t')
rating_df = pd.merge(rating_id_df, name_id_df, on='tconst')

#drop unnecessary columns 
rating_df.drop(['originalTitle','isAdult','endYear','runtimeMinutes','genres'], axis=1, inplace=True)

#loading the plot summaries dataset and add headers
plot_summaries_df = pd.read_csv('data/plot_summaries.txt', sep='\t', header=None, 
                names=['wiki_movie_id', 
                        'plot_summary'])
#merging the movie metadata with the rating data on movie name and release year
movies_ratings = pd.merge(movie_metadata_df, rating_df,  on=['movie_name', 'startYear'])
movies_ratings.shape

# printing the types of the merged data 
movies_ratings['titleType'].unique()

#remove any {{ }} from the plot summary text
plot_summaries_df['plot_summary'] = plot_summaries_df['plot_summary'].str.replace(r'\{\{.*?\}\}', '', regex=True)

# remove all summaries with length = 0
plot_summaries_df = plot_summaries_df[plot_summaries_df['plot_summary'].str.len() > 0]

# keeping only movies, delete tv episodes, tv movies, video games, etc.
movies_ratings = movies_ratings[movies_ratings['titleType']=='movie']


# only keep the movies with more than 100 votes on imdb ratings
movies_ratings = movies_ratings[movies_ratings['numVotes']>200]
movies_ratings.shape

#keep movie_metadata_df only with movies that have ratings
movie_metadata_df = movie_metadata_df[movie_metadata_df['freebase_movie_id'].isin(movies_ratings['freebase_movie_id'])]
movie_metadata_df.shape

#keep the summaries of the selected movies 
plot_summaries_df = plot_summaries_df[plot_summaries_df['wiki_movie_id'].isin(movie_metadata_df['wiki_movie_id'])]
print(plot_summaries_df.shape)

#keep movie_metadata_df only with movies that have summaries
movie_metadata_df = movie_metadata_df[movie_metadata_df['wiki_movie_id'].isin(plot_summaries_df['wiki_movie_id'])]
print(movie_metadata_df.shape)

# save the cleaned summary dataset
plot_summaries_df.to_csv('data/plot_summaries_cleaned.csv', index=False)

# Tokenize, encode, and get embeddings
events['Embeddings'] = events['Event Description'].apply(lambda x: encode(x).tolist() if pd.notnull(x) else None)
#save the embeddings of events as a csv file
events.to_csv('data/events_embeddings.csv', index=False)

# add a column to the plot_summaries_df with embedding of the summary
plot_summaries_df['Embeddings'] = plot_summaries_df['plot_summary'].apply(lambda x: encode(x).tolist() if pd.notnull(x) else None)

#save the embeddings of summaries as a csv file
plot_summaries_df.to_csv('data/plot_summaries_embeddings.csv', index=False)