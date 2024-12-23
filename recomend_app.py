# Import Libraries
import pandas as pd # for data structure and manipulation
import numpy as np # for analytics and computing
from sklearn.feature_extraction.text import TfidfVectorizer # for text classification, clustering, and info retrieval
from sklearn.metrics.pairwise import cosine_similarity # for non-linear similarity measure
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import string # for streamlining input information
import streamlit as st # for web app

glassdoor_clean = pd.read_csv('glassdoor_clean.csv', encoding='utf-8')

# Preprocessing for numerical features (years of experience and salary)
numerical_features = ['experience_years', 'python', 'r_script', 'spark', 'aws', 'excel', 'avg_salary']
numerical_transformer = StandardScaler()

# Preprocessing for categorical columns (skills, city, state)
categorical_features = ['city', 'state']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

text_features = 'job_description'
text_transformer = TfidfVectorizer()

# Combine all preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_features)
    ],
    remainder='drop' # Drop other columns not specified in transformers
)

# Apply transformations to your dataset
X = preprocessor.fit_transform(glassdoor_clean)

# Calculate cosine similarity between the jobs
similarity_matrix = cosine_similarity(X)

# Mapping job titles to indices
indices = pd.Series(glassdoor_clean.index, index=glassdoor_clean['job_title']).drop_duplicates()

def get_recommendations(title, similarity_matrix=similarity_matrix):
    idx = indices[title]  # Get the index of the selected job
    sim_scores = list(enumerate(similarity_matrix[idx]))  # Calculate similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
    sim_scores = sim_scores[1:16]  # Get top 15 similar jobs
    job_indices = [i[0] for i in sim_scores]  # Extract job indices
    return glassdoor_clean['jobtitle'].iloc[job_indices]  # Return the recommended job titles
  
st.header('Data Science Jobs Recommender')

# User input for years of experience
experience_years = st.number_input('Enter your years of work experience:', min_value=0, max_value=50, value=1)

# User input for desired salary
desired_salary = st.number_input('Enter your desired salary (in $):', min_value=0, value=50000)

# User input for skills
skills = st.multiselect(
    'Select your skills:',
    options=['Python', 'R_Script', 'Spark', 'AWS', 'Excel']
)

# User input for city and state
city = st.text_input('Enter your preferred city:')
state = st.text_input('Enter your preferred state:')

# Display recommendations on button click
if st.button('Show Recommendation'):
    recommended_jobs = get_recommendations(
        experience_years, 
        desired_salary, 
        skills, 
        city, 
        state
    )
    st.subheader('Recommended Jobs:')
    for job in recommended_jobs:
        st.write(job)
