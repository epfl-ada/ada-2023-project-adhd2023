# Hack IMDb rating

## Abstract

Amidst the ever-evolving landscape of the global film industry, understanding the factors that lead to a film's success has become crucial. With millions of viewers relying on film ratings to guide their viewing choices, it's essential to understand what drives these ratings. This project aims to reveal the parameters that significantly influence a film's rating, using the IMDb database as its primary resource. 
While primary factors such as budget, genre, cast and plot could be applied for a primary analysis, we aim to perform a deeper and more meaningful search on the elements that combined together in the right way, tend to yield a high ranking.
Such an analysis would be a useful tool in the hands of a filmmaker, critic or just a movie enthusiast, seeking to understand in a more meaningful way the demands of the viewers, depending on the needed era, geographical area or genre one is interested in.

## Research questions
Our analysis will put the light on the movie rating. The research questions that we would like to answer are: 
1) What are the parameters contributing to a movie success and how interpretate them.
2) Does real world event related movie, are more likely to achieve great success.
3) Is it possible to create a model in order to predict movie rating using the relevent and spotted parameters


## Additional datasets
All the data can be found in this drive [link]([https://datasets.imdbws.com](https://drive.google.com/drive/folders/1AVWmDZJu_jlTUsLwY5mGE8g8C7LtoTc3?usp=sharing)) :  

•**IMDb Rating Dataset**:
Our analysis integrates the IMDb rating dataset taken from [IMDb Datasets](https://datasets.imdbws.com). This dataset comprises two CSV files: *rating_id.csv:* Contains IMDb rating data and *name_id.csv:* Includes additional information about the movies (name, type, ...).

•**Events Dataset**:
Supplementary data, generated by ChatGPT, is included in the analysis. This dataset captures the most significant events from 1820 to 2014 and is created using the code found in `generate_events.ipynb`.

•**Oscar Dataset**:
This dataset is a collection of data from The Academy Awards Database, containing the winners and nominees of the Oscars from 1927 to 2023. it is taken from [kaggle Oscars Dataset](https://www.kaggle.com/datasets/unanimad/the-oscar-award) 

•**Budget Dataset**:
This dataset contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. we will use this data to only take the budget column and merge it on our data.  it is taken from [kaggle Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv) 
.
## Methods

### 

 **T-tests** : a statistical test used to compare the means of two groups to determine if they are significantly different from each other.

 **Pearson correlation coefficient**: use PCC to determine linear correlation between set of data.

 **Linear regression** :  a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

 **BERT** : BERT Large by Hugging Face is a larger, more complex version of the BERT model, using bidirectional processing for enhanced language understanding.



## Executed timeline

### Step 1: Data Pre-Processing 

•**Movie Metadata**

For our goal, the data was almost clean, we only removed some outliers.

•**Movie Plot Summaries**

Matching algorithm are highly sensitive to outliers in text and our data included undesired words to describe the type of the plot but the were all between brackets so we proceeded to remove them.

•**IMDB Data**

We first merge our metadata with the IMDb dataset based on the `movie_name` and `start_year`. We noticed that the new data included tvshows, short movies and other non significant types of movies so we proceeded to remove them and filtered the movies who had less than 200 reviews since we agreed that they weren't significant in our analysis.

### Step 2: Analysing Metadata Factors 

The movies metadata helped us draw several helpful conclusions about how to increase the IMDb rating of movies. In this analysis we included the **Language Factor** and **Era Factor** in which we compared the rating means, accounting for a 95% confidence interval and ran a T-test, also in **Number Of Votes Factor** we used linear regression to see how the rating changes with the number of votes and in **Runtime Factor** we used Pearson correlation coefficient to find the optimal movie length.


### Step 3: Analysing Plot Factors

The plot of a movie can be a significant factor in determining both its high and low rankings so we dedicated a lot of analysis to reveal how a movie can take advantage of it to climb the IMDb ranking ladder we used the Bert-Large-cased model from Hugging Face to tokenize and create embeddings for the plot summaries and the events description (this model took 20 hours to run :) ). Then we compare the embedding of the summary to every embedding of the events description and assign each movie to an event based on the best similarity score. We agreed that a plot with similarity above 0.77 with an event is considered a real story movie. Movies that are related to real life events seem to have a better rating.


## PLANS FOR MILESTONE 3

  ### Step 1: Adding an analysis on the movie budget in relation with the imdb rating 
We are willing to merge our datasets with the dataset above to extract the movie budget. After that we will analyse the effect of the movie budget on the IMDb rating.

  ### Step 2: Gender and Ethnicity diversity effect on the IMDb rating
We are willing to analyse the effect of diversity in terms of gender and ethnicity on the IMDb score. To do that we will try to create a new metric for every movie that combines the number of ethnicities in a movies and the percentage of male and female. Then analyse this metric effect on the IMDb rating. We also aim to combine this analysis with the isuing country analysis we did to refine the latter.

   ### Step 3: Happy or Sad movie ending effect on the IMDb rating 
We are willing to use an NLP model to classify movie summaries to know if a movie is has a happy ending or a sad ending. Depending in the results we might try web scraping methods to extract the movie endings from Wikipedia. 


  ### Step 4: Creating a machine learning model to predict movie ratings from the significant factors that we analysed
After further investigations of the factors that make a movie have good or bad ratings we will create a model that predicts the IMDb rating based on those inputs. We will be working on deploying our model on the website so that we can write the inputs (to be defined) on the website and run the model on this input to get the IMDb prediction (we could use Cloud Run in google cloud platform or find another way to deploy it).

### Proposed timeline
 .  
├── 26.11.23 - Step 1  
 │  
├── 03.12.23 - step 2  
 │  
├── 10.12.23 - step 3  
 │  
├── 17.12.24 - step 4  
 │  
├── 22.12.22 - Milestone 3 deadline  
 │  
 .  


### Organization within team

<table class="tg" style="table-layout: fixed; width: 342px">
<colgroup>
<col style="width: 16px">
<col style="width: 180px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0lax">Teammate Name</th>
    <th class="tg-0lax">Teammate Username</th>
    <th class="tg-0lax">Contributions</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">Ali Ridha Mrad </td>
    <td class="tg-0lax">@Ali-Mrad </td>
    <td class="tg-0lax"> Trend analysis<br>Statistic analysis<br>Creation of meaningful visualization <br> Data story development<br> Team management </td>
  </tr>
  <tr>
    <td class="tg-0lax">Aziz Laadhar </td>
    <td class="tg-0lax">@azizlaadhar </td>
    <td class="tg-0lax">Data cleaning<br>Statistic analysis<br> Creation of meaningful visualization <br> Prediction model <br> World event matching</td>
  </tr>
  <tr>
    <td class="tg-0lax">Mohamed Charfi </td>
    <td class="tg-0lax">@charfimohamed </td>
    <td class="tg-0lax">Creation of event dataset <br> Data cleaning<br> Map creation<br>Creation of meaningful visualization<br>World event matching </td>
  </tr>
  <tr>
    <td class="tg-0lax">Nikolay Mikhalev </td>
    <td class="tg-0lax">@moteloumka </td>
    <td class="tg-0lax">Creation of interactive visualization <br> NLP analysis <br> Statistic analysis <br> Trend analysis </td>
  </tr>
  <tr>
    <td class="tg-0lax">Yanis Seddik </td>
    <td class="tg-0lax">@yanvow </td>
    <td class="tg-0lax">Statistic analysis <br>Creation of interactive visualization <br> Network analysis and actor impact <br> Ethnicity analysis <br> Trend analysis </td>
  </tr>
</tbody>
</table>

#### References
