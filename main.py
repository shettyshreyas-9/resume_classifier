import pathlib
import sys
import os
import joblib

import pandas as pd
import numpy as np
import re
import string

import nltk

import streamlit as st



def get_latest_best_model_path(evaluate_log_file):
    with open(evaluate_log_file,'r') as log_file:
        lines= log_file.readlines()
        for line in reversed(lines):
            if 'Best model path' in line:
                return line.split('=')[-1].strip()
            else:
                return 'unavailable'
            
def clean_data(data_df):
    # resume_text= data_df['Resume']

    # remove urls
    cleaned_df= re.sub('http\S+','',data_df)
    # remove RT & cc
    cleaned_df= re.sub('http\S+','',cleaned_df)
    # remove hashtags
    cleaned_df= re.sub('#\S+','',cleaned_df)
    # remove mentions
    cleaned_df= re.sub('@\S+','',cleaned_df)
    # remove non-ASCII character
    cleaned_df= re.sub(r'[^\x00-\x7f]',r' ', cleaned_df)
    # remove extra whitespace
    cleaned_df = re.sub('\s+', ' ', cleaned_df)

    return cleaned_df


def preprocess_data(cleaned_df):

    # Lower case
    preprocessed_df= cleaned_df.lower()

    # Word tokenization
    preprocessed_df= nltk.word_tokenize(preprocessed_df)

    # Removing special characters
    j =[]
    for i in preprocessed_df:
        if i.isalnum():
            j.append(i)

    # Removing stopwords
    k= []
    from nltk.corpus import stopwords
    # stop_words= working_dir
    eng_stop_words= stopwords.words('english')
    punctuations= string.punctuation

    # print(eng_stop_words,'\n')
    # print(punctuations)

    for i in j:
        if i not in eng_stop_words and i not in punctuations:
            k.append(i)


    # Stemming  ** Problem in downloading wordnet so removing lemmetization **
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    l=[]
    for i in k:
        l.append(lemmatizer.lemmatize(i))

    # converting tokens into str 
    preprocessed_df = " ".join(l)
    return preprocessed_df


def main():

    curr_dir = pathlib.Path(__file__)
    working_dir = pathlib.Path(os.getcwd())
    model_path = working_dir.as_posix() + sys.argv[3]

    # Add the directory where you downloaded nltk data to paths for fetching
    nltk.data.path.append( working_dir.as_posix()+ sys.argv[2])
    print(nltk.data.path)


    evaluate_log_file= working_dir.as_posix()+sys.argv[1]
    print(evaluate_log_file)

    best_model_path= get_latest_best_model_path(evaluate_log_file)
    print(best_model_path)

    # Check if the model file exists
    if os.path.exists(best_model_path):

        #Load the best model
        best_model= joblib.load(best_model_path)

    st.title('Resume Classifier')
        
    # getting category & encoded values from train data
    train_df = pd.read_csv(working_dir.as_posix()+ sys.argv[4])

    unique_categories= train_df[['Category','Category_encoded']].drop_duplicates().sort_values('Category_encoded')

    category_dict= dict(zip(unique_categories['Category_encoded'],unique_categories['Category']))
        

    demo_text= '''

Shreyas Shetty 
Data Analyst 
Summary: Data Analyst with expertise in data analysis, machine learning, and data visualization, experienced in driving data-driven solutions for business 
optimization and user engagement. Proven track record in delivering impactful projects across diverse industries. 
 
 shettyshre7@gmail.com                  8999428138                      Pune, India 
                https://www.linkedin.com/in/shreyas-shetty-21ab70151/                               https:/ /shettyshreyas-9.github.io/DA_Portfolio/index.html 
 
Work Experience 
Springer Nature, Pune  
Data Analyst 
April 2022 – Present 
1. Product Performance Analytics: 
a. Developed and maintained the Performance Analytics tool to 
track the performance of product (publication specific) 
portfolios, enabling the editorial, business, and sales teams to 
achieve their targets.  
b. The project involved data pipelining, transformation, and data 
visualization. I leveraged SQL and GBQ for data handling and 
utilized Python for data preprocessing. Tools such as Dataform 
and Looker Studio were employed for data pipelining and data 
visualization. 
 
2. Market Intelligence Tool: 
a. Engineered a versatile intelligence dashboard for market 
analysis across diverse business sectors.  
b. The project encompassed data retrieval, cleaning, and 
preprocessing. We employed SQL and GBQ for data extraction 
and migration, and Python was used for data preprocessing. 
We utilized Looker Studio for data visualization. 
3. In-House ETL and Cost Efficiency: 
a. Orchestrated the shift of ETL management in-house, resulting 
in substantial cost savings. The project involved data migration 
and transformation using SQL and GBQ.  
b. We also leveraged GCP (Google Cloud Platform) for data 
extraction and storage, achieving a significant reduction in data 
discrepancies by 50%. 
4. Article Categorization Automation: 
a. Masterminded an innovative model for automated article 
categorization. The project included data retrieval using SQL, 
data preprocessing with Python, and the development of 
machine learning models.  
b. We employed feature importance, NLP, and Scikit-Learn ML 
algorithms. Our technology stack included Python for data 
preprocessing and Scikit-Learn for machine learning. 
Achievements - Achieved a remarkable 40% increase in user engagement. - Enhanced decision-making accuracy by 20%. - Reduced data discrepancies by 35%. - Successfully implemented decision tree models, saving 25% 
in cleaning time. - Successfully shifted ETL management in-house, saving 
approximately 60,000 Euros. - Achieved a 25% reduction in data processing time. - Created a data visualization solution that increased 
stakeholder engagement by 30%. - Led a data pipelining initiative, reducing data processing time 
by 25%. - Optimized data extraction processes, leading to a 35% 
reduction in data inconsistencies. - Improved subject classification accuracy by 40%. 
   
 
Fittr, Pune  
Product Developer 
October 2019 - January 2021 
1. Smart Fitness Product: 
a. Led the development of a touch screen interactive fitness solution, with 
a strong focus on enhancing user engagement and providing real-time 
insights.  
b. The project involved comprehensive user behavior data collection, 
market research, data analysis, and user profiling. We leveraged 
recommendation engines and predictive modeling. Our technology stack 
included Python, Scikit-Learn, SQL, and Excel. Achievements included a 
remarkable 40% increase in user engagement and improved user 
retention by 30% through recommendation engines. 
 
Futuring Design, Pune 
Product Developer 
September 2018 - September 2019 
1. Agri-tech Product: 
a. Led the development of a data-driven poultry cleaning optimization 
system, incorporating data analysis, machine learning, and data 
visualization.  
b. The project encompassed extensive data collection, market research, 
data cleansing, and preprocessing. We utilized decision tree models and 
linear regression for data analysis. Our technology stack included 
Python, Jupyter Notebook, SQL, and Excel. Notably, we successfully 
implemented decision tree models, resulting in a 25% reduction in 
cleaning time. 
 
SKILLS -  Programming Languages: Python (NumPy, Pandas, Scikit-Learn) -  Machine Learning: Regression, Classification, Clustering -  Data Visualization: Matplotlib, Seaborn, Google Looker Studio -  Database: SQL, MongoDB -  ETL: Google Dataform -  Cloud Platforms: Google Cloud Platform (GCP) -  Tools: Jupyter Notebook, Git, Docker -  Statistical Analysis: Hypothesis Testing, A/B Testing 
 
 
Projects 
1. STEM Salaries &Jobs 
Cleaned & Transformed data using MS Excel. Performed 
analytical queries using MS SQL. Drew insights regarding 
distribution of capital as salary for jobs & the quantitative split 
of job numbers across various domains like job type, region, 
employee demographics etc. Finally created a report 
displaying the results. 
 
2. Political party donations (India) 
Analyzing the distribution of monetary donations given to 
political parties in India by various donors for a decade. 
Created a dashboard to visualize the change in trend of 
donation amount. 
 
3. Covid-19 statistics 
Performed ETL functions on the dataset using SQL. Inferences 
based on the statistics of deaths, death percent, infection 
percent etc were calculated as per area demographics. 
Visualized the following output using Tableau.  
 


        '''
    
    # Get user input text
    sample_text = st.text_input("Enter Resume text:")
    
    cleaned_data= clean_data(sample_text)

    preprocessed_data = preprocess_data(cleaned_data)

    # print(cleaned_data)
    # print(preprocessed_data)

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = joblib.load(model_path+'/vectorizer/tfidf_vectorizer.joblib')

    vectorized_data = tfidf_vectorizer.transform([preprocessed_data])
    print(vectorized_data.shape)

    print(category_dict)

    prediction =best_model.predict(vectorized_data)
    # print(prediction)

    st.write("Resume belongs to category: ", category_dict[prediction[0]])

if __name__=='__main__':
    main()
