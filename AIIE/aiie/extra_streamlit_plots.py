import numpy as np
import pandas as pd
import re
import streamlit as st
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE



st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data  # Use @st.cache_data instead of @st.experimental_memo
def load_data():
    df = pd.read_csv(r"C:\Users\Sofia\PycharmProjects\Incidents\processed_dataset.csv").dropna(how="all")
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



# Replace the slider with a dropdown for selecting the number of top frequencies
top_n_options = ["Top 10", "Top 20", "Top 50", "All"]
frequency_option = st.selectbox("Select the number of top frequencies to display:", options=top_n_options)

# Mapping selected option to a value
top_n_values = {'Top 10': 10, 'Top 20': 20, 'Top 50': 50, 'All': 'All'}
top_n = top_n_values[frequency_option]

# Continue with your logic using the selected `top_n` value


df_processed, technology_set = preprocess_column(df, 'Technology(ies)', 'tech', merge_rename_operations_technology)
df_processed, sector_set = preprocess_column(df_processed, 'Sector(s)', 'sector', merge_rename_operations_sector)
df_processed, issue_set = preprocess_column(df_processed, 'Issue(s)', 'issue', merge_rename_operations_issue)
df_processed, transparency_set = preprocess_column(df_processed, 'Transparency', 'transp', merge_rename_operations_transparency)

# Extracting technology and issue columns with dynamic top N filtering
tech_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('tech_')], top_n)
issue_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('issue_')], top_n)
sector_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('sector_')], top_n)
transp_columns = get_top_columns(df_processed, [col for col in df_processed.columns if col.startswith('transp_')], top_n)



def generate_interactive_heatmap(df_processed, index_columns, column_columns, title, xaxis_title, yaxis_title):
    # Initialize the occurrence matrix with zeros
    occurrence_matrix = pd.DataFrame(0, index=index_columns, columns=column_columns)

    # Calculate occurrences
    for index_col in index_columns:
        for column_col in column_columns:
            if index_col in df_processed.columns and column_col in df_processed.columns:  # Ensure columns exist
                occurrence_matrix.loc[index_col, column_col] = np.logical_and(df_processed[index_col] == 1, df_processed[column_col] == 1).sum()

    # Dynamic adjustments for figure size based on category count
    min_width_per_column = 80  # Minimum width per column
    min_height_per_row = 35  # Minimum height per row
    base_width = 300  # Base width to start with
    base_height = 200  # Base height to start with

    num_columns = len(column_columns)
    num_rows = len(index_columns)
    fig_width = base_width + num_columns * min_width_per_column
    fig_height = base_height + num_rows * min_height_per_row

    # Custom colorscale from white to blue
    custom_colorscale = [[0, 'white'], [1, 'blue']]

    # Create the heatmap with plotly
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
        xaxis_title=xaxis_title,  # Set x-axis title
        yaxis_title=yaxis_title,  # Set y-axis title
    )
    fig.update_xaxes(tickangle=-45)

    return fig



# Continue with your app logic...
option = st.selectbox(
    'Choose the heatmap you want to display',
    ('Technology vs Issue', 'Technology vs Sector', 'Sector vs Issue', 'Technology vs Transparency', 'Issue vs Transparency', 'Sector vs Transparency')
)

# After selecting the heatmap to display
title_mapping = {
    'Technology vs Issue': 'Heatmap of Technology(ies) vs Issue(s)',
    'Technology vs Sector': 'Heatmap of Technology(ies) vs Sector(s)',
    'Sector vs Issue': 'Heatmap of Sector(s) vs Issue(s)',
    'Technology vs Transparency': 'Heatmap of Technology(ies) vs Transparency',
    'Issue vs Transparency': 'Heatmap of Issue(s) vs Transparency',
    'Sector vs Transparency': 'Heatmap of Sector(s) vs Transparency',
}

# Display the selected heatmap's title
st.write(f"### {title_mapping[option]}")
# Add space after the title
st.markdown("<br><br>", unsafe_allow_html=True)

# Mapping option to function call
if option == 'Technology vs Issue':
    fig = generate_interactive_heatmap(
        df_processed=df_processed,
        index_columns=tech_columns,
        column_columns=issue_columns,
        title='',
        xaxis_title="Issue(s)",
        yaxis_title="Technology(ies)"
    )
elif option == 'Technology vs Sector':
    fig = generate_interactive_heatmap(
        df_processed=df_processed,
        index_columns=tech_columns,
        column_columns=sector_columns,
        title='',
        xaxis_title="Sector(s)",
        yaxis_title="Technology(ies)"
    )
elif option == 'Sector vs Issue':
    fig = generate_interactive_heatmap(
        df_processed=df_processed,
        index_columns=sector_columns,
        column_columns=issue_columns,
        title='',
        xaxis_title="Issue(s)",
        yaxis_title="Sector(s)"
    )
elif option == 'Technology vs Transparency':
    fig = generate_interactive_heatmap(
        df_processed=df_processed,
        index_columns=tech_columns,
        column_columns=transp_columns,
        title='',
        xaxis_title="Transparency",
        yaxis_title="Technology(ies)"
    )
elif option == 'Issue vs Transparency':
    fig = generate_interactive_heatmap(
        df_processed=df_processed,
        index_columns=issue_columns,
        column_columns=transp_columns,
        title='',
        xaxis_title="Transparency",
        yaxis_title="Issue(s)"
    )
elif option == 'Sector vs Transparency':
    fig = generate_interactive_heatmap(
        df_processed=df_processed,
        index_columns=sector_columns,
        column_columns=transp_columns,
        title='',
        xaxis_title="Transparency",
        yaxis_title="Sector(s)"
    )

st.plotly_chart(fig, use_container_width=False)




















st.title('Clustering of Incidents (TSNE)')

# Filter for incidents only if your dataset includes various types
df_incidents = df_processed[df_processed['Type'] == 'Incident'].reset_index(drop=True)

# Prepare separate feature categories
feature_categories = {
    'tech': 'Technology Features',
    'transp': 'Transparency Features',
    'issue': 'Issue Features',
    'sector': 'Sector Features'
}
feature_selection = {}

# Helper function to prepare feature selection multiselects
def prepare_feature_multiselects(df, prefix, label):
    features = [col for col in df.columns if col.startswith(prefix)]
    feature_names = [col.replace(prefix + '_', '') for col in features]
    # Include "All" option alongside the feature names
    all_options = ['All'] + feature_names
    # Use a unique key for each multiselect and include the "All" option
    selected_features = st.multiselect(
        label=f'Select {label}:',
        options=all_options,
        default='All',
        key=f'multiselect_{prefix}'  # Maintain unique key for each multiselect
    )
    if 'All' in selected_features:
        # If "All" is selected, return all features with prefix
        return [prefix + '_' + feature for feature in feature_names]
    else:
        # Else, return only the selected features with prefix
        return [prefix + '_' + feature for feature in selected_features]



# Create multiselects for each category and collect selected features
for prefix, label in feature_categories.items():
    selected_features = prepare_feature_multiselects(df_incidents, prefix, label)
    feature_selection[prefix] = selected_features

# Flatten selected features from all categories
all_selected_features = [feature for features in feature_selection.values() for feature in features]


# Apply t-SNE and KMeans clustering if features are selected
if all_selected_features and st.button('Generate Clustering with t-SNE'):
    # Preprocess features with one-hot encoding and standard scaling
    df_features = pd.get_dummies(df_incidents[all_selected_features], drop_first=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(scaled_features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)

    # Define the color mapping for clusters
    color_discrete_map = {
        '0': 'red',
        '1': 'green',
        '2': 'blue',
        '3': 'orange',
        '4': 'purple'
        # Add more colors if there are more than 5 clusters
    }

    # Continue with the rest of your code for t-SNE clustering
    clusters = kmeans.fit_predict(embedding)
    df_embedding = pd.DataFrame(embedding, columns=['t-SNE-1', 't-SNE-2'])
    df_embedding['Cluster'] = clusters.astype(str)  # Convert cluster numbers to string


    def aggregate_features(df, prefix):
        """Aggregate the one-hot encoded features back into a list of features for each document."""
        aggregated_info = []
        for _, row in df.iterrows():
            features = [col.replace(prefix + '_', '') for col in df.columns if col.startswith(prefix) and row[col] == 1]
            aggregated_info.append(", ".join(features))
        return aggregated_info


    # Add Headline/title and aggregated features for hover information
    df_embedding['Headline/title'] = df_incidents['Headline/title'].values
    df_embedding['Technology'] = aggregate_features(df_incidents, 'tech')
    df_embedding['Sector'] = aggregate_features(df_incidents, 'sector')
    df_embedding['Issue'] = aggregate_features(df_incidents, 'issue')
    df_embedding['Transparency'] = aggregate_features(df_incidents, 'transp')

    # Generate the scatter plot
    fig = px.scatter(
        df_embedding,
        x='t-SNE-1',
        y='t-SNE-2',
        color='Cluster',
        hover_data={
            't-SNE-1': False,  # Exclude t-SNE-1 from hover data
            't-SNE-2': False,  # Exclude t-SNE-2 from hover data
            'Headline/title': True,
            'Technology': True,
            'Sector': True,
            'Issue': True,
            'Transparency': True
        },
        color_discrete_map=color_discrete_map  # Apply the color mapping
    )

    fig.update_traces(marker=dict(size=5))

    fig.update_layout(
        title='Incident Clustering with t-SNE & Cluster Coloring',
        margin=dict(l=0, r=0, b=0, t=30),
        legend_title_text='Cluster',
        legend=dict(
            itemsizing='constant',
            title_font_size=25,
            font_size=24
        )
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Please select features and click 'Generate Clustering' to visualize.")