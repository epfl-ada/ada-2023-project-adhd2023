import ast
import time
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from SPARQLWrapper import SPARQLWrapper, JSON
from itertools import combinations

def json_to_values(json_text):
    """
    Convert a JSON string to a list of values.

    Parameters:
    json_text (str): The JSON string to be converted.

    Returns:
    list: A list of values extracted from the JSON string.
          Returns None if the JSON string is invalid.
    """
    
    try:
        json_data = json.loads(json_text)
        values_list = list(json_data.values())
        return values_list
    except json.JSONDecodeError:
        return None
    
def json_to_list_values(df, column_name):
    """
    Convert a column containing JSON strings to a list of values.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to be converted.
        column_name (str): The name of the column to be converted.

    Returns:
        pandas.DataFrame: A copy of the input DataFrame with the specified column converted to a list of values.
    """
    
    df_cop = df.copy()
    df_cop[column_name] = df_cop[column_name].apply(json_to_values)
    return df_cop

def get_wikidata_id(freebase_id):
  """
  Given a Freebase ID, returns the corresponding Wikidata ID and label in English.

  Parameters:
  freebase_id (str): The Freebase ID to look up.

  Returns:
  str: The label of the corresponding Wikidata ID in English, or None if no match is found.
  """


  sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
  query = f"""
  SELECT ?article ?label
  WHERE 
  {{
    ?article schema:about ?item;
        schema:isPartOf <https://en.wikipedia.org/> .
    ?item wdt:P646 "{freebase_id}";
          rdfs:label ?label.
    FILTER(LANG(?label) = "en") # Optional: Filter by English labels
  }}
  """
  sparql.setQuery(query)
  sparql.setReturnFormat(JSON)
  
  try:
      results = sparql.query().convert()
      if 'results' in results and 'bindings' in results['results'] and len(results['results']['bindings']) > 0:
          return results['results']['bindings'][0]['label']['value']
  except Exception as e:
      print(f"Error: {e}")
      time.sleep(20)
      return get_wikidata_id(freebase_id)
  
  return None

def calculate_similarity(a, b):
    """
    Calculates the cosine similarity between two vectors.

    Parameters:
    a (array-like): The first vector.
    b (array-like): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """

    similarity = 1 - cosine(a, b)
    return similarity

def match_event_from_summary_embeddings(events_embedded, movie_embedding, movie_release_year):
    """
    Given a movie embedding and its release year, finds the event that is most similar to the movie based on the embeddings of their summaries.

    Parameters:
    movie_embedding (torch.Tensor): The BERT embedding of the movie summary.
    movie_release_year (str): The year the movie was released.

    Returns:
    tuple: A tuple containing the similarity score between the movie and the most similar event, and the name of the most similar event.
    """

    # Filter events that happened before the movie was released (at least 2 years since it takes time to make a movie)
    filtered_events = \
        events_embedded[events_embedded['Year'] < (int(movie_release_year)-2)]
    # Calculate the similarity between the movie and all events
    similarities = \
        filtered_events['Embeddings'].apply(lambda x: calculate_similarity(movie_embedding, x))
    # Get the index of the most similar event
    index = similarities.idxmax()
    # Get the similarity score of the most similar event
    similarity = similarities[index]
    # Get the name of the most similar event
    matched_event_name = filtered_events.loc[index]['Event Name']
    return similarity, matched_event_name

def plot_data(start, movies_ratings, nb_combined_genres=1, nb_genres=10):
    """
    Plots a histogram of the number of movies produced per genre in a given time period, along with the mean rating of each genre.

    Parameters:
    start (int): The start year of the time period to consider.
    movies_ratings (pandas.DataFrame): A DataFrame containing the movie ratings in column averageRating and genres in column movie_genres.
    nb_combined_genres (int): The number of genres to consider together when finding popular combinations of genres. Default is 1.
    nb_genres (int): The number of genres to include in the plot. Default is 10.

    Returns:
    None
    """

    end = start + 10
    
    df = json_to_list_values(movies_ratings, 'movie_genres')
    #filter df to keep the rows with startYear < end and > start
    df = df[(df['movie_release_year'].astype(int)<end) & (df['movie_release_year'].astype(int)>start)]
    
    if nb_combined_genres > 1:
    #for finding popular combinations of genres
        df['movie_genres'] = df['movie_genres'].apply(lambda x: list(combinations(x, nb_combined_genres)))
        df['movie_genres'] = df['movie_genres'].apply(lambda x: [', '.join(i) for i in x])

    df = df.explode('movie_genres')
    df = df[['averageRating', 'movie_genres']]
    df['count'] = df.groupby('movie_genres')['movie_genres'].transform('count')
    df = df.groupby(['movie_genres']).mean()
    #add a 'count' column to the dataframe
    df = df.sort_values(by=['count'], ascending=False)
    #plot a hisstogram of the first 10 genres with the most movies 
    
    plt.figure(figsize=(16, 3))
    plt.bar(df.index[:nb_genres], df['count'][:nb_genres], width=0.5, color='skyblue')
    #print the mean rating of the first nb_genres genres on the histogram 
    for i, genre in enumerate(df.index[:nb_genres]):
        mean_rating = df.loc[genre, 'averageRating']
        plt.text(i, df['count'][i] + 1, f'rating: {mean_rating:.2f}', ha='center', va='bottom')

    plt.xlabel('Genre')
    plt.ylabel('Number of Movies prodduced')
    plt.title('Number of Movies per genre produdced in the period {}-{}'.format(start, end))
    plt.show()

def t_test_on_language(language_to_test, df, elim_English=False):
    """
    Performs a two-sample t-test to compare the mean rating of movies in a given language to the mean rating of movies in all other languages.

    Parameters:
    language_to_test (str): The name of the language to test.
    df (pandas.DataFrame): A DataFrame containing the movie ratings in column averageRating and languages in column language_name.
    elim_English (bool): Whether to exclude English language films from the analysis. Default is False.

    Returns:
    statistic (float): The t-statistic of the test.
    p_value (float): The p-value of the test.
    """

    if elim_English:
        df = df[df['language_name'] != 'English Language'].copy()
        if language_to_test == 'English Language':
            print("English films have been removed, so no test can be performed")
            return
   # Split the data into two groups: a and all others
    group_a = df[df['language_name'] == language_to_test]['averageRating']
    group_others = df[df['language_name'] != language_to_test]['averageRating']
    # Perform the t-test
    statistic, p_value = stats.ttest_ind(group_a, group_others)  
    return statistic, p_value

# commented out because the kernel crashes when using torch
def encode(text, max_length=512):
    """
    Encodes the given text into a BERT embedding.

    Parameters:
    text (str): The text to encode.
    max_length (int): The maximum length of the input sequence. Defaults to 512.

    Returns:
    torch.Tensor: The BERT embedding of the input text.
    

    # Subtract 2 for [CLS] and [SEP] tokens
    if len(text) == 0:
        print("Empty text")  # Debugging

    tokenizer = BertTokenizer.from_pretrained('bert-Large-cased')
    model = BertModel.from_pretrained('bert-large-cased')
    
    max_length -= 2
    tokens = tokenizer.tokenize(text)
    if len(tokens) == 0:
        print("Empty tokens")  # Debugging
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
            last_hidden_states = model(input_tensor)[0]  # Get the embeddings
        chunk_embeddings.append(last_hidden_states[0].mean(dim=0))

    # Aggregate the embeddings from each chunk 
    embeddings = torch.mean(torch.stack(chunk_embeddings), dim=0)
    return embeddings"""

def string_to_tensor(string):
    """
    Convert a string representation of a tensor back to a tensor object.

    Parameters:
    string (str): The string representation of the tensor.

    Returns:
    torch.Tensor or None: The tensor object if the conversion is successful, None otherwise.
    """
    return None
    """try:
        return torch.tensor(ast.literal_eval(string))
    except ValueError:
        return None"""
    
def plot_mean_for_languages(languages_ratings_df, threshold, top_n=10):
    """
    Plots the mean rating for languages with more than a certain number of films in the dataset, along with a 95% confidence interval.

    Parameters:
    languages_ratings_df (pandas.DataFrame): A DataFrame containing the movie ratings in column averageRating and languages in column language_name.
    threshold (int): The minimum number of films in a language required to be included in the plot.
    top_n (int): The number of languages to include in the plot.

    Returns:
    None
    """
    
    df = languages_ratings_df.copy()
    # calculate the mean and confidence interval of the ratings for languages with more than 500 films and plot them
    df = df.groupby('language_name').filter(lambda x: len(x) > threshold)
    df = df.groupby('language_name')['averageRating'].agg(['mean', 
                                                           'count', 
                                                           'std'])
    df = df.sort_values(by='mean', ascending=False)
    #leave only the first n languages
    df = df[:top_n]

    plt.figure(figsize=(20,10))
    plt.xticks(rotation=45)
    plt.title(f"Mean Rating for Languages with more than {threshold} films in dataset at a 95% confidence interval")
    # plot the mean rating
    sns.barplot(df, 
                x=df.index, 
                y='mean', 
                color='skyblue', 
                errorbar=('ci', 95))

    # plot the confidence interval
    yerr = stats.t.ppf(1-0.025, df['count']-1) * df['std'] / np.sqrt(df['count'])
    #plt.errorbar(x=df.index, y=df['mean'], yerr=yerr, fmt='none', ecolor='black', capsize=5)
    plt.xlabel('Language')
    plt.ylabel('Mean Rating')
    plt.show()

def plot_p_values(p_values):
    """
    Plots the p-values of two-sample t-tests comparing the mean rating of movies in a given language to the mean rating of movies in all other languages.

    Parameters:
    p_values (dict): A dictionary containing the p-values of the t-tests, with language names as keys and p-values as values.

    Returns:
    None
    """

    # plot the p-values with log scale
    plt.figure(figsize=(20,10))
    plt.xticks(rotation=45)
    plt.title('T-Test P-Values for Each Language Compared to All Others (Including English)')
    sns.barplot(x=list(p_values.keys()), y=list(p_values.values()), color='skyblue', log=True)
    plt.axhline(y=0.05, color='red', linestyle='--')
    plt.text(0, 0.05, 'Significance Level', ha='right', va='bottom', color='red', fontsize=14, fontweight='bold')
    plt.xlabel('Language')
    plt.ylabel('P-Value')
    plt.show()

def SDI(x):
    """
    Calculate the Shannon Diversity Index (SDI) of a list of values.

    Parameters:
    -----------
    x : array-like
        A list or array containing numeric values representing the frequencies of different categories.

    Returns:
    --------
    float
        The Shannon Diversity Index value.
    """
    N = x.sum()
    sdi = 0
    for i in x:
        if i > 0:
            freq = i / N
            sdi += -freq * np.log(freq)
    return sdi