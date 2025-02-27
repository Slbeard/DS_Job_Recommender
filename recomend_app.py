import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Read the dataset
glassdoor_clean = pd.read_csv('glassdoor_clean.csv', encoding='utf-8')

# Preprocessing for numerical features
numerical_features = ['experience_years', 'python', 'r_script', 'spark', 'aws', 'excel', 'avg_salary']
numerical_transformer = StandardScaler()

# Preprocessing for categorical columns
categorical_features = ['city', 'state']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine numerical and categorical transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Apply transformations to numerical and categorical features
X_numeric_cat = preprocessor.fit_transform(glassdoor_clean)

# Separate text feature processing
text_transformer = TfidfVectorizer()
X_text = text_transformer.fit_transform(glassdoor_clean['job_description'].fillna(''))

# Combine all transformed features
X = np.hstack([X_numeric_cat.toarray(), X_text.toarray()])

# Compute similarity matrix
similarity_matrix = cosine_similarity(X)

def get_recommendations(experience_years, desired_salary, skills, city, state):
    filtered_jobs = glassdoor_clean.copy()

    # Filter jobs based on experience and salary
    filtered_jobs = filtered_jobs[
        (filtered_jobs['experience_years'] <= experience_years) &
        (filtered_jobs['avg_salary'] >= desired_salary)
    ]
    
    # Normalize skill column names for filtering
    skills = [skill.lower() for skill in skills]
    available_skills = [col for col in skills if col in filtered_jobs.columns]
    
    if available_skills:
        for skill in available_skills:
            filtered_jobs = filtered_jobs[filtered_jobs[skill] == 1]

    # Filter by city and state
    if city:
        filtered_jobs = filtered_jobs[filtered_jobs['city'].str.contains(city, case=False, na=False)]
    if state:
        filtered_jobs = filtered_jobs[filtered_jobs['state'].str.contains(state, case=False, na=False)]

    if filtered_jobs.empty:
        return pd.DataFrame()  # Return empty if no matches

    # Get indices of filtered jobs
    filtered_indices = filtered_jobs.index.tolist()

    # Calculate similarity for filtered jobs
    similarity_scores = similarity_matrix[filtered_indices].mean(axis=0)
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Ensure index mapping is valid
    sorted_indices = [idx for idx in sorted_indices if idx in filtered_jobs.index][:15]

    return filtered_jobs.loc[sorted_indices, ['job_title', 'company', 'avg_salary', 'experience_years']]

# Streamlit App
st.header('Data Science Jobs Recommender')

experience_years = st.number_input('Enter your years of experience:', min_value=0, max_value=50, value=1)
desired_salary = st.number_input('Enter your desired salary (in thousands):', min_value=0, value=50)

skills = st.multiselect('Select your skills:', options=['Python', 'R_Script', 'Spark', 'AWS', 'Excel'])
city = st.text_input('Enter your preferred city:')
state = st.text_input('Enter your preferred state:')

if st.button('Show Recommendation'):
    recommended_jobs = get_recommendations(experience_years, desired_salary, skills, city, state)

    if recommended_jobs.empty:
        st.write("No matching jobs found.")
    else:
        for _, job in recommended_jobs.iterrows():
            st.write(f"**{job['job_title']}** at **{job['company']}**")
            st.write(f"💰 **Salary:** ${job['avg_salary']:.0f}K | 🎓 **Experience Required:** {job['experience_years']} years")
            st.write("---")
