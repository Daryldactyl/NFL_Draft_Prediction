import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import plotly.graph_objects as go
import numpy as np

@st.cache_data
def import_data():
    df = pd.read_csv('data/final_model_data.csv')
    df['Drafted'] = df['Drafted'].astype(int)

    x = df.drop(columns='Drafted')
    columns = list(x.columns)

    y = df['Drafted']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=True, random_state=42)

    sc = StandardScaler()

    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, sc, columns

@st.cache_resource
def train_best_model(_x_train, _y_train, _x_test, _y_test):
    model = RandomForestClassifier(
        bootstrap=False,
        max_depth=30,
        max_features='log2',
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=200,
        random_state=42
    )

    model.fit(_x_train, _y_train)

    #Get accuracy
    y_pred = model.predict(_x_test)

    #Evaluate
    cr = classification_report(_y_test, y_pred, output_dict=True)
    acc = accuracy_score(_y_test, y_pred)

    return model, cr, acc

def create_input(year, position, height, weight, dash, vertical, bench, jump, cone, shuttle, sc):
    input_data = {
        'Year': [year],
        'Weight': [weight],
        '40yd': [dash],
        'Vertical': [vertical],
        'Bench': [bench],
        'Broad Jump': [jump],
        '3Cone': [cone],
        'Shuttle': [shuttle],
        'Height_dnp': [0],
        'Weight_dnp': [0],
        '40yd_dnp': [0],
        'Vertical_dnp': [0],
        'Bench_dnp': [0],
        'Broad Jump_dnp': [0],
        '3Cone_dnp': [0],
        'Shuttle_dnp': [0],
        'Pos_C': [0],
        'Pos_CB': [0],
        'Pos_DB': [0],
        'Pos_DE': [0],
        'Pos_DL': [0],
        'Pos_DT': [0],
        'Pos_EDGE': [0],
        'Pos_FB': [0],
        'Pos_ILB': [0],
        'Pos_K': [0],
        'Pos_LB': [0],
        'Pos_LS': [0],
        'Pos_OG': [0],
        'Pos_OL': [0],
        'Pos_OLB': [0],
        'Pos_OT': [0],
        'Pos_P': [0],
        'Pos_QB': [0],
        'Pos_RB': [0],
        'Pos_S': [0],
        'Pos_TE': [0],
        'Pos_WR': [0],
        'Height_in_inches': [height]
    }

    input_data[f'Pos_{position}'] = [1]
    input_df = pd.DataFrame(input_data)
    input_df_scaled = sc.transform(input_df)

    return input_df_scaled

def make_pred(input_data, model):
    classes = {0: 'Undrafted', 1: 'Drafted'}
    pred = model.predict_proba(input_data)
    pred_class = classes[pred[0].argmax()]
    proba = pred[0][pred[0].argmax()]

    return pred_class, proba


# def compare_position(position, height, weight, dash, vertical, bench, jump, cone, shuttle, sc):
#     # Read the data
#     df = pd.read_csv('data/drafted_player_averages_per_year.csv')
#     position_df = df[df['Position'] == position]
#     numeric_columns = ['Height_in_inches', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle']
#
#     # Create a dictionary with the user's stats
#     user_stats = {
#         'Height_in_inches': height,
#         'Weight': weight,
#         '40yd': dash,
#         'Vertical': vertical,
#         'Bench': bench,
#         'Broad Jump': jump,
#         '3Cone': cone,
#         'Shuttle': shuttle
#     }
#
#     # Calculate mean stats for each position
#     position_stats_mean = position_df.groupby('Position')[numeric_columns].mean().reset_index()
#
#     # Define positions for bar chart
#     positions = numeric_columns
#
#     # Create traces for user stats and player stats
#     trace_user = go.Bar(
#         x=positions,
#         y=[user_stats.get(stat, 0) for stat in positions],  # Get user stats or 0 if not available
#         name='User Stats',
#         marker_color='rgba(50, 171, 96, 0.6)'
#     )
#
#     trace_player = go.Bar(
#         x=positions,
#         y=position_stats_mean.iloc[0][numeric_columns],
#         # Assuming position_stats_mean has only one row for the position
#         name=f'Drafted {position} Stats',
#         marker_color='rgba(96, 50, 171, 0.6)'
#     )
#
#     # Create layout
#     layout = go.Layout(
#         title=f'Comparison with Average Drafted {position}',
#         xaxis=dict(title='Statistic'),
#         yaxis=dict(title='Value')
#     )
#
#     # Create figure
#     fig = go.Figure(data=[trace_user, trace_player], layout=layout)
#
#     return fig
def scale_data(position, height, weight, dash, vertical, bench, jump, cone, shuttle):
    df = pd.read_csv('data/drafted_player_averages_per_year.csv')
    numeric_columns = ['Height_in_inches', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle']
    position_stats = df[df['Position'] == position]

    user_stats = {
        'Height_in_inches': height,
        'Weight': weight,
        '40yd': dash,
        'Vertical': vertical,
        'Bench': bench,
        'Broad Jump': jump,
        '3Cone': cone,
        'Shuttle': shuttle
    }

    position_stats_mean = position_stats[numeric_columns].mean()
    scaled_player = {}
    scaled_users = {}
    for column in numeric_columns:
        max_value = position_stats[column].max()
        min_value = position_stats[column].min()
        scaled_value = (position_stats_mean[column] - min_value)/(max_value-min_value)
        scaled_player[column] = scaled_value
        scaled_user = (user_stats[column] - min_value)/(max_value-min_value)
        scaled_users[column] = scaled_user
    return scaled_player, scaled_users

def compare_position(position, height, weight, dash, vertical, bench, jump, cone, shuttle):
    # Read the data
    df = pd.read_csv('data/drafted_player_averages_per_year.csv')
    numeric_columns = ['Height_in_inches', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle']

    # Filter the data for the given position
    position_stats = df[df['Position'] == position]

    # Calculate mean stats for the filtered position
    position_stats_mean = position_stats[numeric_columns].mean()

    user_stats = {
        'Height_in_inches': height,
        'Weight': weight,
        '40yd': dash,
        'Vertical': vertical,
        'Bench': bench,
        'Broad Jump': jump,
        '3Cone': cone,
        'Shuttle': shuttle
    }

    # Scale the data using scale_data function
    scaled_player, scaled_user = scale_data(position, height, weight, dash, vertical, bench, jump, cone, shuttle)

    # Define positions for bar chart
    positions = numeric_columns

    # Create traces for user stats and player stats
    trace_user = go.Bar(
        x=positions,
        y=list(scaled_user.values()),  # Scaled user stats
        name='User Stats (Scaled)',
        marker_color='rgba(50, 171, 96, 0.6)',
        hoverinfo='text',  # Show hover text
        text=[f'Original: {user_stats[stat]:.2f}' for stat in scaled_user.keys()]  # Original user stats for hover
    )

    trace_player = go.Bar(
        x=positions,
        y=list(scaled_player.values()),  # Scaled player stats
        name=f'{position} Player Stats (Scaled)',
        marker_color='rgba(96, 50, 171, 0.6)',
        hoverinfo='text',  # Show hover text
        text=[f'Original: {position_stats_mean[stat]:.2f}' for stat in scaled_player.keys()]  # Original player stats for hover
    )

    # Create layout
    layout = go.Layout(
        # title=f'Comparison with Average Drafted {position}',
        xaxis=dict(title='Statistics'),
        yaxis=dict(title='Scaled Value'),
        width=900,
        height=600
    )

    # Create figure
    fig = go.Figure(data=[trace_user, trace_player], layout=layout)

    return fig
