import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.subplots as ps
import scipy.stats as stats

def get_similarity_score(vec1, vec2):
    """
    Calculates the similarity score between two vectors.

    Parameters:
    vec1 (array-like): The first vector.
    vec2 (array-like): The second vector.

    Returns:
    float: The similarity score between the two vectors.
    """
    
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def pair_match(treatment_df, movie_linked_year, control_df):
    """
    Creates a bipartite graph and adds edges between treatment_df and control_df based on movie release year and genre similarity.

    Parameters:
    treatment_df (pandas.DataFrame): DataFrame containing treatment data.
    movie_linked_year (pandas.DataFrame): DataFrame containing movie release year data.
    control_df (pandas.DataFrame): DataFrame containing control data.

    Returns:
    nx.Graph: Bipartite graph with edges between treatment_df and control_df.
    """

    G = nx.Graph()
    p = 0
    total_iterations = len(treatment_df.index) 
    for wiki_movie_id1 in treatment_df.index:
        for wiki_movie_id2 in control_df.index:
            if movie_linked_year.loc[wiki_movie_id1,'movie_release_year'] == movie_linked_year.loc[wiki_movie_id2,'movie_release_year']:
                G.add_edge(wiki_movie_id1, wiki_movie_id2, weight=get_similarity_score(treatment_df.loc[wiki_movie_id1,'genres'], control_df.loc[wiki_movie_id2,'genres']))
        p += 1
        print(f'progress: {p}/{total_iterations}')

    return G

def show_graph_pair_matching(treatment_df, control_df):
    """
    Display a graph showing the rating and mean difference between two groups over time.

    Parameters:
    - treatment_df (pandas.DataFrame): DataFrame containing data for the treatment group.
    - control_df (pandas.DataFrame): DataFrame containing data for the control group.

    Returns:
    None
    """
    
    treated_mean = treatment_df.groupby('movie_release_year')['rating'].mean().values.flatten()
    control_mean = control_df.groupby('movie_release_year')['rating'].mean().values.flatten()

    # Confidence interval for treated per year
    treated_std = treatment_df.groupby('movie_release_year')['rating'].std().values.flatten()
    treated_count = treatment_df.groupby('movie_release_year')['rating'].count().values.flatten()

    treated_ci_upper = treated_mean + stats.t.ppf(0.975, treated_count - 1) * treated_std / np.sqrt(treated_count)
    treated_ci_lower = treated_mean - stats.t.ppf(0.975, treated_count - 1) * treated_std / np.sqrt(treated_count)

    # Confidence interval for control per year
    control_std = control_df.groupby('movie_release_year')['rating'].std().values.flatten()
    control_count = control_df.groupby('movie_release_year')['rating'].count().values.flatten()

    control_ci_upper = control_mean + stats.t.ppf(0.975, control_count - 1) * control_std / np.sqrt(control_count)
    control_ci_lower = control_mean - stats.t.ppf(0.975, control_count - 1) * control_std / np.sqrt(control_count)

    # Confidence interval for mean difference per year
    mean_diffs = treated_mean - control_mean
    mean_diffs_std = np.sqrt(treated_std**2 / treated_count + control_std**2 / control_count)

    mean_diffs_ci_upper = mean_diffs + stats.t.ppf(0.975, treated_count + control_count - 2) * mean_diffs_std * np.sqrt(1/treated_count + 1/control_count)
    mean_diffs_ci_lower = mean_diffs - stats.t.ppf(0.975, treated_count + control_count - 2) * mean_diffs_std * np.sqrt(1/treated_count + 1/control_count)

    # Create figure
    fig = ps.make_subplots(
        rows=1,
        cols=2,
        start_cell="bottom-left",
        subplot_titles=("Rating", "Mean Difference (Related - Unrelated)"),
        horizontal_spacing=0.2,
    )

    # Add traces, one for each slider step
    for idx in range(len(treated_mean)):
        fig.add_trace(
            go.Bar(
                visible=False,
                name="Rating",
                x=["Related", "Unrelated"],
                y=[treated_mean[idx], control_mean[idx]],
                error_y=dict(
                    type="data",
                    array=[treated_ci_upper[idx], control_ci_upper[idx]],
                    arrayminus=[treated_ci_lower[idx], control_ci_lower[idx]],
                ),
                marker_color=["#636EFA", "#FFA15A"],
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    for idx, threshold in enumerate(np.arange(1930, 2020, 10)):
        fig.add_trace(
            go.Scatter(
                visible=True,
                name=str(round(mean_diffs[idx])) + " points",
                line=dict(color="#FB0D0D", width=15),
                x=[threshold],
                y=[mean_diffs[idx]],
                mode="markers",
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            visible=True,
            line=dict(color="rgba(47, 138, 196, 1)", width=1),
            x=np.arange(1930, 2020, 10),
            y=mean_diffs,
            name="Mean Difference",
            mode="lines",
        ),
        row=1,
        col=2,
    )

    # add 95% CI
    fig.add_trace(
        go.Scatter(
            visible=True,
            x=np.concatenate([np.arange(1930, 2020, 10), np.arange(1930, 2020, 10)[::-1]]),
            y=np.concatenate([mean_diffs_ci_upper, mean_diffs_ci_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(47, 138, 196, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% CI",
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # Make traces visible
    fig.data[len(treated_mean) - 1].visible = True
    fig.data[len(treated_mean) * 2 - 1].visible = True


    # Create and add slider
    steps = []
    for idx, year in enumerate(np.arange(1930, 2020, 10)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=str(year),
        )
        step["args"][0]["visible"][min(idx, len(treated_mean) - 1)] = True
        step["args"][0]["visible"][min(idx + len(treated_mean), 2 * len(treated_mean) - 1)] = True
        step["args"][0]["visible"][min(2 * len(treated_mean), len(fig.data) - 1)] = True
        steps.append(step)

    sliders = [
        dict(
            active=10,
            currentvalue={"prefix": "Movies after year: "},
            pad={"t": 50},
            steps=steps,
            step=dict(step=10)  # set the step size to 10 years
        )
    ]

    # edit axis labels
    fig["layout"]["yaxis"]["title"] = "Mean box office revenue"
    fig["layout"]["yaxis2"]["title"] = "Mean difference"
    fig["layout"]["xaxis"]["title"] = "Group"
    fig["layout"]["xaxis2"]["title"] = "Year"

    sliders = [
            dict(
                active=50,
                currentvalue={"prefix": "Threshold on ethnicity score (%): "},
                pad={"t": 50},
                steps=steps,
            )
        ]

    fig.update_layout(width=700, height=500, sliders=sliders)

    fig.show()
