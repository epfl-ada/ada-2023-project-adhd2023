"""
Originally this was a notebook that created the labels for the movies plots.
"""

import pandas as pd
from transformers import pipeline

df = pd.read_csv('/kaggle/input/cmu-things/plot_summaries.txt', delimiter='\t', encoding='utf-8', names=['wiki_movie_id', 'wiki_plot'])

ratings = pd.read_csv('/kaggle/input/cmu-things/movies_metadata_ratings.csv')
ratings = ratings[['averageRating','numVotes','wiki_movie_id','movie_name','movie_release_year']]


merged_df = df.merge(ratings,on = 'wiki_movie_id' , how='inner')
df_sorted = merged_df.sort_values(by='averageRating', ascending=False)
df = df_sorted.set_index('wiki_movie_id')
df = df[['wiki_plot','averageRating','numVotes','movie_name','movie_release_year']]

top_10 = df.head(100).copy()
modern_films_df = df[ df['movie_release_year'] > 1980 ].copy()
modern_films_df = modern_films_df[modern_films_df['numVotes'] > 1000]

modern_films_df.shape

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",device =0)

# List 2: Emotional Tone-Based Classes
emotion_classes = [
    "Heartwarming",
    "Suspenseful",
    "Inspiring",
    "Dark",
    "Humorous",
    "Touching",
    "Tense",
    "Mysterious",
    "Thrilling",
    "Light-hearted"
]

# List 3: Setting/Time Period-Based Classes
setting_classes = [
    "Historical",
    "Contemporary",
    "Futuristic",
    "Period Piece",
    "Post-Apocalyptic",
    "Fantasy World",
    "Urban",
    "Rural",
    "Space",
    "War"
]

# List 4: Character-Centric Classes
character_classes = [
    "Hero's Journey",
    "Antihero",
    "Ensemble Cast",
    "Coming-of-Age",
    "Protagonist vs. Antagonist",
    "Lone Survivor",
    "Buddy Film",
   # "Family-Centered",
    "Villain Origin",
    "Strong Female Lead"
]

# List 5: Plot Structure-Based Classes
structure_classes = [
    "Linear Narrative",
    "Non-linear Narrative",
    "Flashbacks",
    "Multiple Timelines",
    "Parallel Storylines",
    "Plot Twist",
    "Redemption Arc",
    "Quest",
    "Tragedy",
    "Slice of Life"
]

# List 6: Location-Based Classes
location_classes = [
    "City Life",
    "Small Town",
    "Desert",
    "Island",
    "Mountains",
    "Underwater",
    "Space Exploration",
    "Suburbia",
    "Wilderness",
    "International"
]

# List 7: Source Material-Based Classes
source_material_classes = [
    "Based on a Novel",
    "Adaptation of a True Story",
    "Remake",
    "Original Screenplay",
    "Comic Book Adaptation",
    "Video Game Adaptation",
    "Fairy Tale Retelling",
    "Historical Event",
    "Biographical",
    "Mythology-Based"
]

# List 8: Plot Themes-Based Classes
theme_classes = [
    "Love and Romance",
    "Revenge",
    "Identity and Self-discovery",
    "Survival",
    "Betrayal",
    "Justice and Morality",
    "Redemption",
    "Power and Corruption",
    "Friendship",
    "Quest for Knowledge"
]
# List 9: Plot Motif-Based Classes
plot_motif_classes = [
    "Hero's Journey",
    "Rags to Riches",
    "The Quest",
    "Overcoming the Monster",
    "Tragedy",
    "Comedy",
    "Voyage and Return",
    "Rebirth",
    "The Underdog",
    "Forbidden Love"
]

# List 10: Conflict-Based Classes
conflict_classes = [
    "Man vs. Nature",
    "Man vs. Self",
    "Man vs. Society",
    "Man vs. Machine",
    "Man vs. Supernatural",
    "Man vs. Fate",
    "Man vs. Technology",
    "Man vs. Alien",
    "Man vs. God",
    "Man vs. Time"
]

# List 11: Mood-Based Classes
mood_classes = [
    "Dark and Gritty",
    "Whimsical",
    "Inspirational",
    "Cerebral",
    "Surreal",
    "Melancholic",
    "Epic",
    "Satirical",
    "Hopeful",
    "Eerie"
]

# List 12: Relationship-Based Classes
relationship_classes = [
    "Family Dynamics",
    "Friendship and Loyalty",
    "Forbidden Love",
    "Bitter Rivalry",
    "Mentorship",
    "Parent-Child Bond",
    "Siblings",
    "Love Triangle",
    "Complicated Relationships",
    "Unlikely Friendships"
]

# List 13: Action Sequence-Based Classes
action_sequence_classes = [
    "Epic Battles",
    "Car Chases",
    "Hand-to-Hand Combat",
    "Chase Scenes",
    "Explosions",
    "Heist",
    "Escape",
    "Sword Fights",
    "Gunfights",
    "High-Speed Pursuit"
]

# List 14: Time Period-Based Classes
time_period_classes = [
    "Medieval",
    "Renaissance",
    "Victorian Era",
    "20th century",
    "Present day",
    "Future",
    "Alternate History"
]

# List 15: Nature-Based Classes
nature_classes = [
    "Ocean Exploration",
    "Jungle Adventures",
    "Survival in the Wild",
    "Mountain Climbing",
    "Underground Exploration",
    "Desert Survival",
    "Arctic Expeditions",
    "African Safari",
    "Island Paradise",
    "Amazon Rainforest"
]

# List 16: Political Themes-Based Classes
political_themes_classes = [
    "Political Corruption",
    "Revolution",
    "Injustice",
    "Government Conspiracy",
    "Civil Rights",
    "War Crimes",
    "Espionage",
    "Global Politics",
    "Tyranny",
    "Uprising"
]

# Combine all lists into one list
all_lists = [
    emotion_classes, setting_classes, character_classes,
    structure_classes, location_classes, source_material_classes, theme_classes,
    plot_motif_classes, conflict_classes, mood_classes, relationship_classes,
    action_sequence_classes, time_period_classes, nature_classes, political_themes_classes
]

# Drama-Specific Lists
family_drama_classes = [
    "Parent-Child Relationships",
    "Siblings' Rivalry",
    "Generational Conflict",
    "Marital Struggles",
    "Reconciliation",
    "Loss and Grief",
    "Coming of Age",
    "Family Secrets",
    "Addiction and Recovery",
    "Forgiveness"
]

social_issues_drama_classes = [
    "Racial Discrimination",
    "Social Injustice",
    "Economic Disparity",
    "Gender Inequality",
    "LGBTQ+ Themes",
    "Mental Health",
    "Immigration",
    "Bullying",
    "Poverty",
    "Aging and Elderly Issues"
]

workplace_drama_classes = [
    "Corporate Politics",
    "Work-Life Balance",
    "Office Romance",
    "Harassment and Discrimination",
    "Ambition and Success",
    "Ethical Dilemmas",
    "Professional Rivalry",
    "Job Loss and Unemployment",
    "Burnout",
    "Entrepreneurship"
]

period_drama_settings_classes = [
    "Victorian Era",
    "Roaring Twenties",
    "Great Depression",
    "World War I",
    "World War II",
    "Civil Rights Era",
    "1960s",
    "1970s",
    "Historical Scandals",
    "Renaissance"
]


# Lists of Tropes and Genres
old_hollywood_tropes = [
    "The Femme Fatale",
    "The Swashbuckler",
    "The Hard-Boiled Detective",
    "The Musical Number",
    "The Screwball Comedy",
    "The Gangster",
    "The Gentleman Thief",
    "The Ingenue",
    "The Wise Mentor",
    "The Grand Romantic Gesture"
]

silent_film_tropes = [
    "The Damsel in Distress",
    "Slapstick Comedy",
    "Title Cards",
    "Melodrama",
    "The Keystone Kops",
    "Chase Scenes",
    "The Villain Tied to Railroad Tracks",
    "The Mysterious Stranger",
    "The Strongman",
    "The Expressionist Set Design"
]

classic_film_genres = [
    "Film Noir",
    "Westerns",
    "Screwball Comedies",
    "Musicals",
    "War Films",
    "Romantic Comedies",
    "Adventure Films",
    "Epic Historical Dramas",
    "Monster Movies",
    "Gothic Horror"
]

#acton type films

# Action Movie Themes
action_movie_themes = [
    "High-Stakes Heist",
    "Undercover Agent",
    "Martial Arts Showdown",
    "Race Against Time",
    "Vengeance Quest",
    "Terrorist Threat",
    "Rescue Mission",
    "Battle for Survival",
    "Espionage and Intrigue",
    "Superhero Origins"
]

# Iconic Action Sequences
iconic_action_sequences = [
    "Car Chase",
    "Explosive Shootout",
    "Hand-to-Hand Combat",
    "Fistfight on a Moving Vehicle",
    "Chase on Foot",
    "Daring Helicopter Escape",
    "Epic Gun Battle",
    "High-Octane Motorcycle Chase",
    "Building or Rooftop Leap",
    "Submarine Showdown"
]

# Action Movie Archetypes
action_movie_archetypes = [
    "Lone Hero",
    "Rogue Cop",
    "Master Assassin",
    "Fearless Leader",
    "Sidekick or Partner",
    "Femme Fatale",
    "Mercenary or Soldier",
    "Tech Genius",
    "Evil Mastermind",
    "Henchmen or Minions"
]

# Action Movie Settings
action_movie_settings = [
    "Urban Cityscape",
    "Exotic Jungle",
    "Desert Wasteland",
    "Underground Lair",
    "High-Speed Train",
    "Secret Military Base",
    "Post-Apocalyptic World",
    "Ancient Temple Ruins",
    "Remote Island",
    "Space Station or Spaceship"
]

all_action_movie_lists = [
    action_movie_themes,
    iconic_action_sequences,
    action_movie_archetypes,
    action_movie_settings
]


# Concatenate the lists
all_old_movie_tropes_and_genres = [old_hollywood_tropes ,silent_film_tropes ,classic_film_genres]

drama_list = [family_drama_classes,period_drama_settings_classes,workplace_drama_classes,social_issues_drama_classes]
i = 38
sequence_to_classify = df.iloc[i]['wiki_plot']
print(df.iloc[i]['movie_name'])

for candidate_labels in all_lists + drama_list + all_old_movie_tropes_and_genres + all_action_movie_lists :
  output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
  #print(output['labels'],output['scores'])
  for label, score in zip(output['labels'], output['scores']):

        if score > 0.55:
            print(label, score)

def analyser(text):
    """
    Analyzes the given text and returns a dictionary of labels and their scores.

    Args:
        text (str): The text to be analyzed.

    Returns:
        dict: A dictionary containing labels as keys and their corresponding scores as values.
    """
  
    dico = {}
    print("==============")
    for candidate_labels in all_lists + drama_list + all_old_movie_tropes_and_genres + all_action_movie_lists :
        output = classifier(text, candidate_labels, multi_label=False)
        #print(output['labels'],output['scores'])
        for label, score in zip(output['labels'], output['scores']):
            if score > 0.40:
                print(label,score)
                dico[label] = score
    return dico

df = modern_films_df.copy()
df = df.iloc[9500:9909].copy()
df.shape

df['labels'] = df['wiki_plot'].apply(analyser)

df.to_csv('/kaggle/working/9500_to_9909_labels.csv', index=True)

sequence_to_classify = df.iloc[14]['wiki_plot']
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output['labels'],output['scores'])

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ARTICLE = df.iloc[i]['wiki_plot']
summary = summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)
print(summary)