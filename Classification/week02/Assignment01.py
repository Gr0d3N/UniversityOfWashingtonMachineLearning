#!/usr/bin/env python
# coding: utf-8

# In[1]:


# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # Load review dataset

# In[2]:


# importing the data
products = pd.read_csv('amazon_baby_subset.csv')
products.head()

# # Apply text cleaning on the review data

# In[3]:


# important words
import json

with open('important_words.json') as f:
    important_words = json.load(f)
    
print(important_words)

# In[4]:


# missing values
products = products.fillna({'review':''})

# In[5]:


# remove punctuation
import string

def remove_punctuation(text):
    trans = str.maketrans('', '', string.punctuation)
    return text.translate(trans) 

products.review.fillna('', inplace=True)
products['review_clean'] = products['review'].apply(remove_punctuation)
products.head()

# In[6]:


# count of important words
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

products.head()

# ## How many reviews contain the word perfect?

# In[7]:


# number of reviews containing 'perfect'
products['contains_perfect'] = np.where(products['perfect'] >= 1, 1, 0)
products['contains_perfect'].sum()

# # Convert data fram to multi-dimensional array

# In[43]:


def extract_features_labels(dataframe, features, label):
    """
    A function that extracts features, prepends a constant column of value 1, and labels from a dataframe
    param df: pd.DataFrame
    param features: list of features
    param label: string of the label or the target
    :return: 2D array of features, 1D array of class labels
    """
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.values
    label_sarray = dataframe[label]
    label_array = label_sarray.values
    return(feature_matrix, label_array) 


# ## How many features are there in the feature matrix?

# In[58]:


feature_matrix, sentiment = extract_features_labels(products, important_words, 'sentiment')
feature_matrix.shape

# The number of features is 193 assuming the intercept is not a feature

# In[59]:


sentiment.shape

# In[63]:


def predict_probability(feature_matrix, coefficients):
    score = np.dot(feature_matrix, coefficients)
    predictions = 1 / (1 + np.exp(-score))
    return predictions

# In[64]:


def feature_derivative(errors, feature_matrix):
    return np.dot(errors, feature_matrix)

# In[65]:


def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores)))
    return lp

# # Taking gradient steps

# In[71]:


from math import sqrt
def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in range(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = (sentiment == +1)
        errors = indicator - predictions
        
        for j in range(len(coefficients)):
            derivative = np.dot(errors, feature_matrix[:, j])
            coefficients[j] = step_size * derivative

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print('iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp))
    
    return coefficients

# In[72]:


initial_coefficients = np.zeros(194,)
step_size = 1e-7
max_iter = 301

coefficients = logistic_regression(featre_matrix, sentiment, initial_coefficients, step_size, max_iter)
print(coefficients)

# ## As each iteration of gradient ascent passes, does the log likelihood increase or decrease?
# decrease

# # Predicting sentiment

# ## How many reviews were predicted to have positive sentiment?

# In[74]:


scores_new = np.dot(feature_matrix, coefficients)
predicted_sentiment = np.array([+1 if s > 0 else -1 for s in scores_new])
sum(predicted_sentiment == +1)

# # Measuring accuracy

# ## What is the accuracy of the model on predictions made above? (round to 2 digits of accuracy)

# In[79]:


correctly_classified = predicted_sentiment == sentiment
correctly_classified.sum()/len(sentiment)

# # Which words contivute most to positive & negative sentiments

# In[80]:


coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)

# # Ten "most positive" words

# ## Which word is not present in the top 10 "most positive" words?

# In[83]:


word_coefficient_tuples[:10]

# # Ten "most negative" words

# ## Which word is not present in the top 10 "most negative" words?

# In[86]:


word_coefficient_tuples[-10:]

# In[ ]:



