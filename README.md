# Emotion Detection in Text using Natural Language Processing

<img src="./images/index.png"/>

<br>

# Introduction
Emotion detection in text data involves identifying the emotions expressed in textual data. This can be a challenging task since emotions are often expressed in complex and subtle ways. Natural language processing (NLP) techniques can be used to analyze text data and identify the emotions expressed in it.

The aim of this project is to develop a model that uses NLP techniques to accurately detect emotions in text data. The model can be used for sentiment analysis, customer feedback analysis, and social media monitoring. The model is trained on a dataset of text data that has been labeled with the corresponding emotions expressed in it.

# Dataset
The <a src="./data/">dataset</a> used for this project contains text data labeled with one of eight emotions: anger, disgust, fear, joy, neutral, sadness, shame and surprise. The dataset contains a total of 34795 rows.

# Methodology
- The methodology used for this project involves the following steps:
1. Preprocessing the text data: The text data is preprocessed by removing stop words, punctuation, user handles and converting all text to lowercase. 
2. Model training: A machine learning model is trained on the extracted features to predict the emotions expressed in the text data. The model used for this project is a Logistic Regression and MultinomialNB.
3. Model evaluation: The trained model is evaluated on the test data to measure its accuracy in detecting emotions in text data.

# Results
The Logistic Regression achieved an accuracy of 62% on the data.

# Installation
1. Clone the repository to your local machine:
```
https://github.com/SannketNikam/Emotion-Detection-in-Text.git
```

2. Install the 'requirements.txt':
```
pip install -r requirements.txt
```

3. To run this project :
```
streamlit run app.py
```

4. It'll automatically open the Streamlit app in your default browser.
