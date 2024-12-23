import pandas as pd  # for data structure and manipulation
import numpy as np  # for analytics and computing
from sklearn.feature_extraction.text import TfidfVectorizer  # for text classification, clustering, and info retrieval
from sklearn.metrics.pairwise import cosine_similarity  # for non-linear similarity measure
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st  # for web app

# Read and preprocess the data
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
    remainder='drop'  # Drop other columns not specified in transformers
)

# Apply transformations to your dataset
X = preprocessor.fit_transform(glassdoor_clean)

# Calculate cosine similarity between the jobs
similarity_matrix = cosine_similarity(X)

# Mapping job titles to indices
indices = pd.Series(glassdoor_clean.index, index=glassdoor_clean['job_title']).drop_duplicates()

def get_recommendations(experience_years, desired_salary, skills, city, state, similarity_matrix=similarity_matrix):
    # Initial DataFrame for filtering
    filtered_jobs = glassdoor_clean.copy()

    # Filter jobs based on experience and salary
    filtered_jobs = filtered_jobs[
        (filtered_jobs['experience_years'] <= experience_years) &
        (filtered_jobs['avg_salary'] >= desired_salary)
    ]
    
    st.write(f"Filtered jobs after experience and salary: {filtered_jobs.shape[0]} jobs.")

    # Check for matching skills
    if skills:
        for skill in skills:
            filtered_jobs = filtered_jobs[filtered_jobs[skill.lower()] == 1]  # Assuming binary skill columns
        st.write(f"Filtered jobs after skills: {filtered_jobs.shape[0]} jobs.")

    # Filter by city and state (if provided)
    if city:
        filtered_jobs = filtered_jobs[filtered_jobs['city'].str.contains(city, case=False, na=False)]
        st.write(f"Filtered jobs after city filter: {filtered_jobs.shape[0]} jobs.")
    if state:
        filtered_jobs = filtered_jobs[filtered_jobs['state'].str.contains(state, case=False, na=False)]
        st.write(f"Filtered jobs after state filter: {filtered_jobs.shape[0]} jobs.")

    # If no jobs are left after filtering, return an empty result
    if filtered_jobs.empty:
        st.write("No matching jobs found after filtering.")
        return filtered_jobs  # Returning empty DataFrame if no jobs match

    # Get indices of filtered jobs
    filtered_indices = filtered_jobs.index.tolist()

    # Calculate similarity for filtered jobs
    similarity_scores = similarity_matrix[filtered_indices]
    average_similarity = similarity_scores.mean(axis=0)  # Average similarity for each job
    sorted_indices = np.argsort(average_similarity)[::-1]  # Sort by descending similarity

    # Ensure we return no more than the number of available jobs
    top_n = len(sorted_indices)  # Set the top the number of available jobs

    # Safely select the top N recommended jobs
    sorted_indices = sorted_indices[::top_n]  # Only slice the top N
    
    # Use sorted_indices to index filtered_jobs and return the top jobs
    recommended_jobs = filtered_jobs.iloc[sorted_indices]

    return recommended_jobs[['job_title', 'company', 'avg_salary', 'experience_years']]

recommended_jobs = None  # Initialize the variable

# Streamlit app layout
st.header('Data Science Jobs Recommender')

# User input for years of experience
experience_years = st.number_input('Enter your years of work experience:', min_value=0, max_value=50, value=1)

# User input for desired salary (in thousands)
desired_salary = st.number_input('Enter your desired salary in thousands (to the nearest $1K):', min_value=0, value=50000)

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

if recommended_jobs is None or recommended_jobs.empty:
    st.write("No matching jobs found.")
else:
    for _, job in recommended_jobs.iterrows():
        st.write(f"**Job Title:** {job['job_title']}")
        st.write(f"**Company:** {job['company']}")
        st.write(f"**Average Salary:** ${job['avg_salary']:.0f}K")
        st.write(f"**Experience Required:** {job['experience_years']} years")
        st.write("---")
