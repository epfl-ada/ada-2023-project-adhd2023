import pandas as pd
import plotly.graph_objects as go
import ast
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import  mean_absolute_error
from wordcloud import WordCloud
import statsmodels.api as sm

def linear_regression(X_column_names, y_column_names, df):
    """
    Perform linear regression on the given DataFrame.

    Parameters:
    X_column_names (list): List of column names to be used as independent variables.
    y_column_names (str): Name of the column to be used as the dependent variable.
    df (pandas.DataFrame): Input DataFrame containing the data.

    Returns:
    tuple: A tuple containing the trained model, mean absolute error on the training set, and mean absolute error on the test set.
    """

    df_train = df.sample(frac=0.85, random_state=1)
    df_test = df.drop(df_train.index)

    X = sm.add_constant(df_train[X_column_names])
    model = sm.OLS(df_train[y_column_names], X).fit()

    X_test = sm.add_constant(df_test[X_column_names])
    preds = model.predict(X_test)
    mae_train = mean_absolute_error(df_train[y_column_names], model.predict(X))
    mae_test = mean_absolute_error(df_test[y_column_names], preds)
    return model, mae_train, mae_test

def filter_model_information(model, ci_threshold=0.1):
    """
    Filters the model information based on p-value and confidence interval.

    Args:
        model: The regression model object.
        ci_threshold: The threshold for the width of the confidence interval.

    Returns:
        filtered_model_df: The filtered model information dataframe.
    """

    model_df = pd.DataFrame(model.summary().tables[1])

    model_df.columns = model_df.iloc[0]
    model_df.columns = ['Key Word', 'Coefficient', 'Standard Error', 't-value', 'p-value', '95% CI Lower', '95% CI Upper']
    model_df = model_df[1:].copy()

    for col in model_df.columns:
        model_df[col] = model_df[col].apply(lambda x: x.data)
    for col in model_df.columns[1:]:
        model_df[col] = model_df[col].astype(float)
    model_df = model_df[1:].copy()

    filtered_model_df = model_df[model_df['p-value'] <= 0.05]
    filtered_model_df = filtered_model_df[(filtered_model_df['95% CI Upper'] - filtered_model_df['95% CI Lower'] <= ci_threshold) & 
                                          (filtered_model_df['95% CI Lower'] * filtered_model_df['95% CI Upper'] > 0)]
    
    return filtered_model_df


def show_model(model,top_to_show=5, ci_threshold=0.1):
    """
    Display a bar plot of the coefficients influencing the rating.

    Parameters:
    - model: The regression model object.
    - top_to_show: The number of top and bottom coefficients to display. Default is 5.
    - ci_threshold: The confidence interval threshold for filtering coefficients. Default is 0.1.
    """
    
    filtered_model_df = filter_model_information(model, ci_threshold=ci_threshold)
    top = filtered_model_df.nlargest(top_to_show, 'Coefficient')
    bottom = filtered_model_df.nsmallest(top_to_show, 'Coefficient')
    filtered_model_df = pd.concat([top, bottom])

    # Create a Plotly figure with custom error bars
    fig = go.Figure()

    for index, row in filtered_model_df.iterrows():
        genre = row['Key Word']
        coef = row['Coefficient']
        
        # Add a bar trace with custom error bars
        fig.add_trace(go.Bar(
            x=[coef],
            y=[genre],
            orientation='h',
            name=genre
        ))

    # Customize the layout
    fig.update_layout(
        yaxis_title='Key Word',
        xaxis_title='Coefficient Value',
        title='Coefficients influencing the rating',
        barmode='group'  # Use 'group' to display multiple bars per genre
    )

    # Show the plot
    fig.show()

# Custom color functions for gradients
class ColorFuncWithGradient(object):
    """
    A callable class that generates color values based on word frequency.

    Args:
        color_hue (int): The hue value for the color.
        word_freq (dict): A dictionary containing word frequencies.

    Returns:
        str: A color value in HSL format.

    """

    def __init__(self, color_hue, word_freq):
        self.color_hue = color_hue
        self.word_freq = word_freq

    def __call__(self, word, font_size, position, orientation, random_state=None, **kwargs):
        # Normalize coefficient value to a scale of 0 to 1
        normalized_weight = self.word_freq[word] / max(self.word_freq.values())
        lightness = 100 - normalized_weight * 50  # Scale lightness
        lightness = max(0, min(lightness, 100))  # Ensure lightness is between 0 and 100
        return f"hsl({self.color_hue}, 100%, {lightness}%)"

# Function to create and display word cloud
def create_and_display_wordcloud(word_freq, color_hue, title):
    """
    Create and display a word cloud based on the given word frequencies.

    Parameters:
    - word_freq (dict): A dictionary containing word frequencies.
    - color_hue (str): The color hue for the word cloud.
    - title (str): The title of the word cloud.

    Returns:
    None
    """
    wordcloud = WordCloud(width=800, height=800, max_font_size=100, min_font_size=10)
    wordcloud.generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud.recolor(color_func=ColorFuncWithGradient(color_hue, word_freq)), interpolation="bilinear")
    plt.title(title)
    plt.axis("off")
    plt.show()