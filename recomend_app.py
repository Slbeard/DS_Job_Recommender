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

def get_recommendations(experience_years, desired_salary, skills, city, state, similarity_matrix=similarity_matrix):
    # Filter jobs based on experience and salary
    filtered_jobs = glassdoor_clean[
        (glassdoor_clean['experience_years'] <= experience_years) &
        (glassdoor_clean['avg_salary'] >= desired_salary)
    ]
    
    # Check for matching skills
    if skills:
        for skill in skills:
            filtered_jobs = filtered_jobs[filtered_jobs[skill.lower()] == 1]  # Assuming binary skill columns

    # Filter by city and state (if provided)
    if city:
        filtered_jobs = filtered_jobs[filtered_jobs['city'].str.contains(city, case=False, na=False)]
    if state:
        filtered_jobs = filtered_jobs[filtered_jobs['state'].str.contains(state, case=False, na=False)]

    # Get indices of filtered jobs
    filtered_indices = filtered_jobs.index.tolist()

    # Calculate similarity for filtered jobs
    similarity_scores = similarity_matrix[filtered_indices]
    average_similarity = similarity_scores.mean(axis=0)  # Average similarity for each job
    sorted_indices = np.argsort(average_similarity)[::-1]  # Sort by descending similarity

    # Return top 15 recommended job titles
    return glassdoor_clean.iloc[sorted_indices[:15]][['job_title', 'company', 'avg_salary']]
  
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
        experience_years=experience_years,
        desired_salary=desired_salary,
        skills=skills,
        city=city,
        state=state
    )
st.subheader('Recommended Jobs:')
    if recommended_jobs.empty:
        st.write("No matching jobs found.")
    else:
        for _, job in recommended_jobs.iterrows():
            st.write(f"**Job Title:** {job['jobtitle']}")
            st.write(f"**Company:** {job['company']}")
            st.write(f"**Average Salary:** ${job['avg_salary']:,.2f}")
            st.write("---")
