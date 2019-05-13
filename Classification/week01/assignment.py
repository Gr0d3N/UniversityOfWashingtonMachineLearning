
# coding: utf-8

# In[1]:

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import string
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# In[2]:

# loading the dataset
products = pd.read_csv('amazon_baby.csv')
products.head()


# In[3]:

# perform text cleaning
def remove_punctuation(text):
    trans = str.maketrans('', '', string.punctuation)
    return text.translate(trans) 

products.review.fillna('', inplace=True)
products['review_clean'] = products['review'].apply(remove_punctuation)
products.head()


# In[4]:

# extract sentiment
products = products[products.rating != 3]

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products.head()


# In[5]:

# split train test data

# getting the tarin indices
with open('module-2-assignment-train-idx.json') as f:
    train_idx = json.load(f)
    
# getting the test indices
with open('module-2-assignment-test-idx.json') as f:
    test_idx = json.load(f)
    
train_data = products.iloc[train_idx]
test_data = products.iloc[test_idx]


# In[6]:

# build the word count vector for each review

vectorized = CountVectorizer(token_pattern=r'\b\w+\b') # This token pattern to keep single-letter words

# train using train dataset
train_matrix = vectorized.fit_transform(train_data['review_clean'])

# convert the test dataset
test_matrix = vectorized.transform(test_data['review_clean'])


# In[7]:

# train a logistic regression model
sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])


# In[8]:

# How many wiehgts are >= 0?

coeff_array = sentiment_model.coef_
print((coeff_array >= 0).sum())


# In[9]:

# making perdictions with the logistic regression
sample_test_data = test_data[10:13]
print(sample_test_data)


# In[10]:

# digging in the 1st row of the sampel_test_data
sample_test_data.iloc[0]['review']


# In[11]:

# looking at the next row (-ve review)
sample_test_data.iloc[1]['review']


# In[13]:

# calculating the score for the sample_test_data
sample_test_matrix = vectorized.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print(scores)


# In[14]:

# predict sentiment
sentiment_model.predict(sample_test_matrix)


# In[15]:

# probablity predictions
probability = 1/(1 + np.exp(-scores))
probability


# In[16]:

# find the most positive (and negative) review

test_matrix = vectorized.transform(test_data['review_clean'])
scores_test = sentiment_model.decision_function(test_matrix)
predictions = 1/(1+np.exp(-scores_test))

test_data['predictions'] = predictions

test_data.sort_values('predictions', ascending=False).iloc[0:20]


# In[18]:

test_data.sort_values('predictions', ascending = True).iloc[0:20]


# In[19]:

# compute the accuracy of the classifier
predicted_sentiment = sentiment_model.predict(test_matrix)
test_data['predicted_sentiment'] = predicted_sentiment
test_data['diff_sentiment'] = predicted_sentiment - test_data['sentiment']
acc = np.sum(sum(test_data['diff_sentiment'] == 0))
print(acc)


# In[20]:

total = len(test_data.index)
print(total)


# In[22]:

print(round(float(acc)/float(total), 2))


# In[24]:

# learn another classifer with fewer words

significant_words =['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']

vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])


# In[25]:

# train a logistic regression model on a subset of data

simple_model = LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_data['sentiment'])


# In[32]:

simple_model.coef_.flatten()


# In[34]:

simple_model_coef_table = pd.DataFrame({'word':significant_words,
                                        'coefficient':simple_model.coef_.flatten()})


# In[36]:

simple_model_coef_table.sort_values('coefficient', ascending=False)


# In[37]:

sum(simple_model_coef_table['coefficient']>=0)


# In[41]:

vocab = list(vectorized.vocabulary_.keys())
coeffs = {vocab[i]: c for i, c in enumerate(sentiment_model.coef_[0])}
new_dic = {k:v for k, v in coeffs.items() if k in significant_words}
new_table = pd.DataFrame(new_dic.items(), columns=['word', 'new_coeffi'])
new_table_coeff = pd.merge(simple_model_coef_table, new_table, how = 'left', on = 'word' )
new_table_coeff = new_table_coeff[new_table_coeff['coefficient']>=0]
sum(new_table_coeff['new_coeffi'] < 0)


# In[42]:

new_table_coeff


# In[43]:

# comparing models

predicted_sentiment_train_sentiment = sentiment_model.predict(train_matrix)
train_data['predicted_sentiment_ts'] = predicted_sentiment_train_sentiment
acc_ts = round(float(sum(train_data['predicted_sentiment_ts'] == train_data['sentiment']))/len(train_data.index),2)


# In[44]:

predicted_simple_train_sentiment = simple_model.predict(train_matrix_word_subset)
train_data['predicted_simple_ts'] = predicted_simple_train_sentiment
acc_tsimple = round(float(sum(train_data['predicted_simple_ts'] == train_data['sentiment']))/len(train_data.index),2)


# In[45]:

acc_ts > acc_tsimple


# In[46]:

acc_tsimple


# In[47]:

acc_ts


# In[48]:

predicted_sentiment_test_sentiment = sentiment_model.predict(test_matrix)
test_data['predicted_sentiment_ts'] = predicted_sentiment_test_sentiment
acc_ts_test = round(float(sum(test_data['predicted_sentiment_ts'] == test_data['sentiment']))/len(test_data.index),2)
acc_ts_test


# In[49]:

predicted_simple_test_sentiment = simple_model.predict(test_matrix_word_subset)
test_data['predicted_simple_ts'] = predicted_simple_test_sentiment
acc_tsimple_test = round(float(sum(test_data['predicted_simple_ts'] == test_data['sentiment']))/len(test_data.index),2)
acc_tsimple_test


# In[50]:

acc_ts_test > acc_tsimple_test


# In[51]:

# the majority class model

sum(train_data['sentiment'] == 1)


# In[52]:

sum(train_data['sentiment'] == -1)


# In[53]:

round(float(sum(test_data['sentiment'] ==1))/len(test_data.index),2)


# In[ ]:



