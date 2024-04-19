import numpy as np
import re
import streamlit as st
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.cluster import KMeans
import os
import pandas as pd

######################## INCIDENTS HEATMAPS ########################

# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))

csv_path = os.path.join(dir_path, 'processed_dataset.csv')


st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data  # Use @st.cache_data instead of @st.experimental_memo
def load_data():
    df = pd.read_csv(csv_path).dropna(how="all")
    return df


df = load_data()
df_processed = df.copy()

# Function to clean, split, and normalize column values
def clean_split_normalize(values):
    if pd.isna(values):
        return []
    return [value.strip().lower() for value in re.split(';|,', values) if value.strip()]

# Preprocess 'Technology(ies)', 'Sector(s)', and 'Issue(s)' columns
def preprocess_column(df, column_name, prefix, merge_rename_operations={}):
    df[column_name] = df[column_name].str.lstrip()
    unique_set = set()
    df[column_name].apply(lambda x: [unique_set.add(item) for item in clean_split_normalize(x)])
    encoding_dict = {f'{prefix}_{item}': [] for item in unique_set}
    for _, row in df.iterrows():
        items = clean_split_normalize(row[column_name])
        for item in unique_set:
            encoding_dict[f'{prefix}_{item}'].append(int(item in items))
    encoded_df = pd.DataFrame(encoding_dict)
    for new_name, old_names in merge_rename_operations.items():
        encoded_df[new_name] = encoded_df[old_names].max(axis=1)
        encoded_df.drop(columns=old_names, inplace=True, errors='ignore')
    processed_df = pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)
    return processed_df, unique_set



merge_rename_operations_technology = {
    'tech_unclear/unknown ': ['tech_unclear/unknown', 'tech_unknown'],
    'tech_facial recognition/detection/identification ': ['tech_facial recognition', 'tech_facial detection', 'tech_facial recogniton', 'tech_facial detection'],
    'tech_(performance) scoring algorithm ': ['tech_performance scoring algorithm', 'tech_scoring algorithm'],
    'tech_location analytics ': ['tech_location tracking', 'tech_location recognition',  'tech_location analytics'],
    'tech_social media (monitoring) ': ['tech_social media monitoring', 'tech_social media'],
    'tech_emotion recognition/detection ': ['tech_emotion recognition', 'tech_emotion detection'],
    'tech_neural networks ': ['tech_neural networks', 'tech_neural network'],
    'tech_image generation ': ['tech_image generator', 'tech_image generation'],
    'tech_large language model (llm) ': ['tech_large language model', 'tech_large language model (llm)'],
    'tech_speech/voice recognition ': ['tech_speech/voice recognition', 'tech_speech ecognition', 'tech_speech recognition', 'tech_voice recognition'],
    'tech_image recognition/filtering ': ['tech_image recognition/filtering', 'tech_image recognition'],
    'tech_vehicle detection ': ['tech_vehicle detection', 'tech_vehicle detection system'],
    'tech_object recognition/detection/identification ': ['tech_object identification', 'tech_object recognition', 'tech_object detection'],
    'tech_voice generation/synthesis ': ['tech_voice synthesis', 'tech_voice generation'],
    'tech_gaze recognition/detection ': ['tech_gaze detection', 'tech_gaze recognition'],
    'tech_text-to-speech ': ['tech_text-to-speech', 'tech_text to speech'],
    'tech_virtual reality (vr) ': ['tech_virtual reality (vr)', 'tech_virtual reality'],
    'tech_behavioural analysis ': ['tech_behavioural monitoring', 'tech_behavioural monitoring system', 'tech_behavioural analysis',],
    'tech_predictive (statistical) analytics ': ['tech_predictive statistical analysis', 'tech_predictive analytics'],
    'tech_content management/moderation system ': ['tech_content moderation system', 'tech_content management system'],
    'tech_deepfake - audio ': [ 'tech_deepfake - audio', 'tech_audio'],
    'tech_deepfake - image ': [ 'tech_deepfake - image',  'tech_image', ],
    'tech_deepfake - video ': [ 'tech_deepfake - video',  'tech_video'],
    'tech_gesture analysis ': ['tech_gesture analysis', 'tech_gesture recognition',  'tech_smile recognition'],
    'tech_pricing algorithm ': ['tech_pricing algorithm', 'tech_price adjustment algorithm', 'tech_pricing automation'],
    'tech_facial analysis ': ['tech_facial analysis', 'tech_facial matching',  'tech_facial scanning'],
    'tech_fingerprint analysis ': [ 'tech_fingerprint biometrics', 'tech_fingerprint scanning', 'tech_fingerprint recognition'],
    'tech_risk assessment algorithm/system ': [ 'tech_risk assessment algorithm',  'tech_risk assessment/classification algorithm', 'tech_recidivism risk assessment system',  'tech_automated risk assessment'],
    'tech_scheduling algorithm/software ': [ 'tech_scheduling algorithm', 'tech_crew scheduling software'],
}

# Merge and rename specific columns as requested for 'Technology(ies)'
merge_rename_operations_sector = {
    'sector_real estate sales / management': ['sector_real estate sales/management',  'sector_real estate'],
    'sector_govt - health ': [ 'sector_gov - health', 'sector_govt - health'],
    'sector_business/professional services ': [ 'sector_professional/business services',  'sector_business/professional services'],
    'sector_govt - police ': [ 'sector_govt - police', 'sector_police'],
    'sector_govt - agriculture ': [ 'sector_govt - agriculture', 'sector_agriculture'],
    'sector_private - individual ': [ 'sector_private - individual', 'sector_private'],
    'sector_banking/financial services ': [ 'sector_banking/financial services', 'sector_govt - finance'],
    'sector_education ': [ 'sector_education', 'sector_govt - education'],
    'sector_telecoms ': ['sector_telecoms',  'sector_govt - telecoms']
}

merge_rename_operations_issue = {
    'issue_bias/discrimination - lgbtqi+': ['issue_bias/discrimination - transgender',  'issue_transgender',  'issue_bias/discrimination - sexual preference (lgbtq)', 'issue_bias/discrimination - lgbtq', 'issue_lgbtq',],
    'issue_necessity/proportionality ': [ 'issue_necessity/proportionality', 'issue_proportionality'],
    'issue_bias/discrimination - race/ethnicity ': [ 'issue_bias/discimination - race', 'issue_race','issue_bias/disrimination - race','issue_ethnicity','issue_bias/discrimination - racial', 'issue_bias/discrimination - ethnicity',  'issue_bias/discrimination - race',  'issue_bias/disrimination - ethnicity',],
    'issue_bias/discrimination - political ': ['issue_bias/discrimination - politics', 'issue_bias/discrimination - political',  'issue_political'],
    'issue_mis/dis-information ': ['issue_mis-disinformation',  'issue_mis/dsinformation', 'issue_mis/disinformation'],
    'issue_autonomous lethal weapons ': ['issue_autonomous lethal weapons', 'issue_lethal autonomous weapons'],
    'issue_governance/accountability - capability/capacity ': [ 'issue_capability/capacity','issue_governance/accountability - capability/capacity',  'issue_governance/accountability', 'issue_governance - capability/capacity'],
    'issue_ethics/values ': ['issue_ethics', 'issue_ethics/values'],
    'issue_ownership/accountability ': [ 'issue_ownership/accountability',  'issue_accountability'],
    'issue_ip/copyright ': ['issue_copyright','issue_ip/copyright'],
    'issue_reputational damage ': [ 'issue_reputation', 'issue_reputational damage'],
    'issue_bias/discrimination - employment/income ': [ 'issue_bias/discrimination - employment', 'issue_employment', 'issue_bias/discrimination - income', 'issue_income', 'issue_bias/discrimination - profession/job'],
    'issue_legal - liability ': [ 'issue_legal - liability',  'issue_liability'],
    'issue_identity theft/impersonation ': [ 'issue_identity theft/impersonation', 'issue_impersonation'],
    'issue_bias/discrimination - disability ': ['issue_bias/discrimination - disability', 'issue_disability'],
    'issue_bias/discrimination - economic ': [ 'issue_bias/discrimination - economic',  'issue_economic'],
    'issue_bias/discrimination - political opinion/persuasion ': [ 'issue_bias/discrimination - political opinion','issue_bias/discrimination - political persuasion'],
    'issue_nationality ': ['issue_national origin', 'issue_nationality',  'issue_national identity'],
    'issue_employment - pay/compensation ': ['issue_pay', 'issue_employment - pay',  'issue_employment - pay/compensation'],
    'issue_corruption/fraud ': ['issue_fraud', 'issue_corruption/fraud', 'issue_legal - fraud', 'issue_safety - fraud'],
    'issue_bias/discrimination - gender ': [ 'issue_bias/discrimination - gender',  'issue_gender'],
    'issue_accuracy/reliabiity ': [ 'issue_accuracy/reliabiity', 'issue_accuracy/reliability',  'issue_accuracy/reliabilty', 'issue_accuracy/reliablity',  'issue_accuray/reliability'],
    'issue_bias/discrimination - body size/weight ': ['issue_size',  'issue_body size', 'issue_weight',],
    'issue_bias/discrimination - location ': [ 'issue_bias/discrimination - location', 'issue_location'],
    'issue_employment - unionisation ': [ 'issue_unionisation', 'issue_employment - unionisation'],
    'issue_employment - jobs ': ['issue_employment - jobs', 'issue_jobs'],
    'issue_bias/discrimination - religion ': [ 'issue_bias/discrimination - religion', 'issue_religion'],
    'issue_employment - health & safety ': [ 'issue_employment - health & safety',  'issue_employment - safety'],
    'issue_bias/discrimination - age ': [ 'issue_bias/discrimination - age', 'issue_age'],
    'issue_privacy - consent ': ['issue_privacy - consent', 'issue_privacy'],
    'issue_value/effectiveness ': ['issue_value/effectiveness', 'issue_effectiveness/value'],
    'issue_oversight/review ': ['issue_oversight','issue_oversight/review'],
    'issue_legal - defamation/libel ': ['issue_legal - defamation/libel',  'issue_defamation'],
    'issue_misleading marketing ': ['issue_misleading marketing','issue_misleading'],
    'issue_bias/discrimination - education ':['issue_education'],
    'issue_employment - termination ': ['issue_employment - termination','issue_termination'],
    'issue_anthropomorphism ': [ 'issue_robot rights','issue_anthropomorphism'],
    'issue_humanrights_freedom' :['issue_freedom of expression - right of assembly',  'issue_freedom of expression - censorship',  'issue_freedom of expression', 'issue_freedom of information'],

}

merge_rename_operations_transparency = {
    'transp_black box ': [ 'transp_back box', 'transp_black box','transp_governance: black box'],
    'transp_legal ': ['transp_legal', 'transp_legal - mediation', 'transp_legal - foi request blocks'],
    'transp_marketing ': ['transp_marketing: privacy', 'transp_marketing privacy','transp_marketing', 'transp_governance: marketing', 'transp_marketing - hype', 'transp_marketing - misleading'],
    'transp_privacy/consent': ['transp_privacy - consent', 'transp_consent', 'transp_privacy'],
    'transp_complaints/appeals': [ 'transp_complaints/appeals', 'transp_complaints & appeals','transp_appeals/complaints'],
    'transp_existence ': ['transp_existence',  'transp_governance - existence']

}
# Function to get the top N or all columns based on frequency
def get_top_columns(df, columns, top_n):
    frequency = df[columns].sum().sort_values(ascending=False)
    if top_n != 'All':
        return frequency.head(top_n).index.tolist()
    return frequency.index.tolist()


# Streamlit app layout
st.title('Data Exploration Heatmaps')

# Description for the heatmap
st.write("Heatmaps representing the counts between specific pairs of features to check how the two variables (in the x/y axes) are related.")


# Dropdown with radio button for selecting the number of top frequencies
top_n_options = ["Top 10", "Top 20", "Top 50", "All"]
frequency_option = st.radio("Select the number of top frequencies to display:", options=top_n_options)

# Mapping selected option to a value remains the same
top_n_values = {'Top 10': 10, 'Top 20': 20, 'Top 50': 50, 'All': 'All'}
top_n = top_n_values[frequency_option]


df_processed, technology_set = preprocess_column(df, 'Technology(ies)', 'tech', merge_rename_operations_technology)
df_processed, sector_set = preprocess_column(df_processed, 'Sector(s)', 'sector', merge_rename_operations_sector)
df_processed, issue_set = preprocess_column(df_processed, 'Issue(s)', 'issue', merge_rename_operations_issue)
df_processed, transparency_set = preprocess_column(df_processed, 'Transparency', 'transp', merge_rename_operations_transparency)

# Extracting technology and issue columns with dynamic top N filtering
tech_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('tech_')], top_n)
issue_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('issue_')], top_n)
sector_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('sector_')], top_n)
transp_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('transp_')], top_n)



# Available categories for selection
categories = {
    'Technology': 'tech',
    'Sector': 'sector',
    'Issue': 'issue',
    'Transparency': 'transp'
}
# User selects the category for the X axis
x_axis_option = st.selectbox(
    'Choose X axis:',
    options=list(categories.keys()),  # Display the keys for selection
    index=0  # Default selection (first item)
)

# User selects the category for the Y axis
y_axis_option = st.selectbox(
    'Choose Y axis:',
    options=list(categories.keys()),  # Display the keys for selection
    index=1  # Default to second item to avoid same default as X axis
)

# Dynamic title based on user selection
dynamic_title = f"Heatmap of {x_axis_option} vs {y_axis_option}"


# Generate the heatmap based on the user selection
def generate_heatmap_from_selection(df_processed, x_axis_category, y_axis_category):
    # Get the correct columns for each axis based on the user's selection
    x_columns = get_top_columns(df_processed, [col for col in df_processed.columns if
                                               col.startswith(categories[x_axis_category] + '_')], top_n)
    y_columns = get_top_columns(df_processed, [col for col in df_processed.columns if
                                               col.startswith(categories[y_axis_category] + '_')], top_n)

    # Generate and return the interactive heatmap
    return generate_interactive_heatmap(df_processed, x_columns, y_columns, '', x_axis_category, y_axis_category)


# Create the interactive heatmap
def generate_interactive_heatmap(df_processed, x_columns, y_columns, title, xaxis_label, yaxis_label):
    # Initialize the occurrence matrix with zeros
    occurrence_matrix = pd.DataFrame(0, index=y_columns, columns=x_columns)

    # Calculate occurrences
    for y_col in y_columns:
        for x_col in x_columns:
            occurrence_matrix.loc[y_col, x_col] = np.logical_and(
                df_processed[y_col] == 1, df_processed[x_col] == 1).sum()

    # Dynamic adjustments for figure size
    min_width_per_column = 35  # Minimum width per column
    min_height_per_row = 35  # Minimum height per row
    base_width = 300  # Base width to start with
    base_height = base_width  # Base height to start with

    num_columns = len(x_columns)  # Correct variable name used here
    num_rows = len(y_columns)  # Correct variable name used here
    fig_width = base_width + num_columns * min_width_per_column
    fig_height = base_height + num_rows * min_height_per_row

    # Custom colorscale
    custom_colorscale = [[0, 'white'], [1, 'blue']]

    # Create the heatmap with plotly
    # Create the heatmap
    fig = ff.create_annotated_heatmap(
        z=occurrence_matrix.values,
        x=[col.split('_', 1)[1] for col in occurrence_matrix.columns],
        y=[idx.split('_', 1)[1] for idx in occurrence_matrix.index],
        annotation_text=occurrence_matrix.values.astype(str),
        showscale=True,
        colorscale=custom_colorscale,
    )
    fig.update_layout(
        title=title,
        autosize=False,
        width=fig_width,
        height=fig_height,
        margin=dict(t=50, l=50, b=150, r=50),
        xaxis_title=xaxis_label,
        yaxis_title=yaxis_label,
    )
    fig.update_xaxes(tickangle=-45)

    return fig




# After selecting the heatmap to display
title_mapping = {
    'Technology vs Issue': 'Heatmap of Technology(ies) vs Issue(s)',
    'Technology vs Sector': 'Heatmap of Technology(ies) vs Sector(s)',
    'Sector vs Issue': 'Heatmap of Sector(s) vs Issue(s)',
    'Technology vs Transparency': 'Heatmap of Technology(ies) vs Transparency',
    'Issue vs Transparency': 'Heatmap of Issue(s) vs Transparency',
    'Sector vs Transparency': 'Heatmap of Sector(s) vs Transparency',
}


# Generate the heatmap with the corrected axes
heatmap_fig = generate_heatmap_from_selection(df_processed, x_axis_option, y_axis_option)

# Correct the axis titles
corrected_xaxis_title = x_axis_option  # These might need to be swapped to align with the plot's orientation
corrected_yaxis_title = y_axis_option

# Update the layout with the corrected axis titles
heatmap_fig.update_layout(
    xaxis_title=corrected_xaxis_title,
    yaxis_title=corrected_yaxis_title
)

# Display the heatmap with the corrected title and axis labels
st.write(f"### {dynamic_title}")
st.markdown("<br><br>", unsafe_allow_html=True)
with st.container():
    st.plotly_chart(heatmap_fig, use_container_width=True)











######################## INCIDENTS UMAP VISUALIZATION ########################

st.title('Incidents (UMAP)')

# Description for the UMAP visualization
st.write("UMAP visualization that represents incidents based on selected features, allowing for exploration of data similarity.")

# Assuming 'df_processed' also contains a 'Type' column to filter incidents
df_incidents = df_processed[df_processed['Type'] == 'Incident'].reset_index(drop=True)

# Prepare separate feature categories based on column prefixes in 'df_processed'
feature_categories = {
    'tech': 'Technology Features',
    'transp': 'Transparency Features',
    'issue': 'Issue Features',
    'sector': 'Sector Features'
}

# User selects the category for coloring
coloring_options = st.selectbox('Select the category to color by:', options=list(feature_categories.keys()))

# User selects the number of top features to display in the chosen category
top_k = st.slider('Select the number of top categories to display:', min_value=1, max_value=20, value=5)

# Get all features for the selected category
all_features_in_category = [col for col in df_incidents.columns if col.startswith(coloring_options)]
# Set or adjust the number of neighbors
n_neighbors = st.slider('Select the number of neighbors for UMAP:', min_value=5, max_value=50, value=15)


def aggregate_features(df, prefix):
    """Aggregate the one-hot encoded features back into a list of features for each document."""
    aggregated_info = []
    for _, row in df.iterrows():
        features = [col.replace(prefix + '_', '') for col in df.columns if col.startswith(prefix) and row[col] == 1]
        aggregated_info.append(", ".join(features))
    return aggregated_info

# Generate the aggregated feature strings for the dataframe
df_incidents['Technology'] = aggregate_features(df_incidents, 'tech')
df_incidents['Sector'] = aggregate_features(df_incidents, 'sector')
df_incidents['Issue'] = aggregate_features(df_incidents, 'issue')
df_incidents['Transparency'] = aggregate_features(df_incidents, 'transp')

# Process for displaying the UMAP
if st.button('Generate UMAP Visualization'):
    # Get top K columns in the selected category for coloring
    # Get top K columns in the selected category for coloring
    top_k_columns = get_top_columns(df_incidents, all_features_in_category, top_k)

    # Filter incidents containing at least one of the top-k features
    df_filtered_incidents = df_incidents[df_incidents[top_k_columns].any(axis=1)]

    # Make sure to capture the headlines corresponding to the filtered incidents
    headlines = df_filtered_incidents['Headline/title'].values  # Capture the filtered headlines

    # Encode features for UMAP
    df_features = pd.get_dummies(df_filtered_incidents[top_k_columns], drop_first=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    # Perform UMAP
    from umap import UMAP

    umap_instance = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
    embedding = umap_instance.fit_transform(scaled_features)

    # Prepare the DataFrame for plotting
    df_embedding = pd.DataFrame(embedding, columns=['UMAP-1', 'UMAP-2'])
    df_embedding['Category'] = df_filtered_incidents[top_k_columns].idxmax(axis=1).apply(lambda x: x.split('_', 1)[-1])
    df_embedding['Headline/title'] = headlines
    df_embedding['Technology'] = df_filtered_incidents['Technology'].values
    df_embedding['Sector'] = df_filtered_incidents['Sector'].values
    df_embedding['Issue'] = df_filtered_incidents['Issue'].values
    df_embedding['Transparency'] = df_filtered_incidents['Transparency'].values

    # Create plot with hover data
    fig = px.scatter(
        df_embedding,
        x='UMAP-1',
        y='UMAP-2',
        color='Category',
        hover_data={
            'UMAP-1': False,
            'UMAP-2': False,
            'Headline/title': True,
            'Technology': True,
            'Sector': True,
            'Issue': True,
            'Transparency': True
        },
        title='Incident Visualization with UMAP colored by ' + feature_categories[coloring_options],
        width=700, height=700
    )

    # Update plot appearance
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))

    with st.container():
        st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Please select a category and click 'Generate UMAP Visualization' to visualize.")
