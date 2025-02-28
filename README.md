# Data Science Jobs Recommender  

## Overview  
This project is a **job recommendation system** for data science roles. It leverages a structured **Glassdoor dataset** to match job seekers with relevant job postings based on their **experience, skills, location, and salary expectations**.  

The system is built using **Streamlit**, **Scikit-learn**, and **Pandas**, applying **TF-IDF vectorization** and **cosine similarity** to recommend the most relevant job listings.  

## Features  
- **Job Filtering:** Filters jobs based on experience, salary, skills, and location.  
- **Text-based Similarity:** Uses **TF-IDF** to analyze job descriptions.  
- **Categorical and Numerical Processing:** Applies **StandardScaler** and **OneHotEncoder** for preprocessing.  
- **User-friendly UI:** Built with **Streamlit** for easy job searching.  

## Installation  
Ensure you have Python 3 installed, then install dependencies:  
```bash
pip install pandas numpy streamlit scikit-learn
```  

## Running the Application  
1. **Place** `glassdoor_clean.csv` in the project directory.  
2. **Run** the Streamlit app:  
   ```bash
   streamlit run app.py
   ```  

## How It Works  
1. **Preprocessing:**  
   - Numerical features (e.g., experience, salary) are **standardized**.  
   - Categorical features (city, state) are **one-hot encoded**.  
   - Job descriptions are **vectorized using TF-IDF**.  
2. **Recommendation:**  
   - Computes **cosine similarity** between job listings.  
   - Filters by user’s inputs (experience, salary, skills, location).  
   - Returns **top job recommendations** based on similarity scores.  
3. **UI Interaction:**  
   - Users input preferences (experience, salary, skills, location).  
   - Click “Show Recommendation” to display top jobs.  

## Future Improvements  
- **Real-time job postings** via APIs.  
- **More job sources** beyond Glassdoor.  
- **Advanced ML models** for better recommendations.  

## Dataset  
Sourced from **Glassdoor (via Kaggle)**, containing:  
- Job details (title, description, salary, company, location)  
- Skills requirements (Python, R, Spark, AWS, Excel)  
- Categorical data (state, industry, ownership type)  

## Acknowledgments  
Dataset: [Glassdoor Data Science Jobs (Kaggle)](https://www.kaggle.com/datasets/kuralamuthan300/glassdoor-data-science-jobs)  
