
# coding: utf-8

# # Exploring precision and recall
# 
# The goal of this second notebook is to understand precision-recall in the context of classifiers.
# 
#  * Use Amazon review data in its entirety.
#  * Train a logistic regression model.
#  * Explore various evaluation metrics: accuracy, confusion matrix, precision, recall.
#  * Explore how various metrics can be combined to produce a cost of making an error.
#  * Explore precision and recall curves.
#  
# Because we are using the full Amazon review dataset (not a subset of words or reviews), in this assignment we return to using GraphLab Create for its efficiency. As usual, let's start by **firing up GraphLab Create**.
# 
# Make sure you have the latest version of GraphLab Create (1.8.3 or later). If you don't find the decision tree module, then you would need to upgrade graphlab-create using
# 
# ```
#    pip install graphlab-create --upgrade
# ```
# See [this page](https://dato.com/download/) for detailed instructions on upgrading.

# # Load amazon review dataset

# In[2]:

import pandas as pd
import numpy as np
products = pd.read_csv('amazon_baby.csv')
products.head()


# # Extract word counts and sentiments

# As in the first assignment of this course, we compute the word counts for individual words and extract positive and negative sentiments from ratings. To summarize, we perform the following:
# 
# 1. Remove punctuation.
# 2. Remove reviews with "neutral" sentiment (rating 3).
# 3. Set reviews with rating 4 or more to be positive and those with 2 or less to be negative.

# In[3]:

# remove punctuation
import string

def remove_punctuation(text):
    trans = str.maketrans('', '', string.punctuation)
    return text.translate(trans) 

products.review.fillna('', inplace=True)
products['review_clean'] = products['review'].apply(remove_punctuation)
products.head()


# In[4]:

products = products[products['rating'] != 3]


# In[5]:

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)


# Now, let's remember what the dataset looks like by taking a quick peek:

# In[6]:

products


# ## Split data into training and test sets
# 
# We split the data into a 80-20 split where 80% is in the training set and 20% is in the test set.

# In[9]:

import json
with open('module-9-assignment-train-idx.json', 'r') as f: # Reads the list of most frequent words
    train_idx = json.load(f)
with open('module-9-assignment-test-idx.json', 'r') as f1: # Reads the list of most frequent words
    test_idx = json.load(f1)
    
train_data = products.iloc[train_idx]
test_data = products.iloc[test_idx]


# In[10]:

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])


# ## Train a logistic regression classifier
# 
# We will now train a logistic regression classifier with **sentiment** as the target and **word_count** as the features. We will set `validation_set=None` to make sure everyone gets exactly the same results.  
# 
# Remember, even though we now know how to implement logistic regression, we will use GraphLab Create for its efficiency at processing this Amazon dataset in its entirety.  The focus of this assignment is instead on the topic of precision and recall.

# In[14]:

from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression


# In[15]:

model = LogisticRegression()


# In[16]:

model.fit(train_matrix, train_data['sentiment'])


# # Model Evaluation

# We will explore the advanced model evaluation concepts that were discussed in the lectures.
# 
# ## Accuracy
# 
# One performance metric we will use for our more advanced exploration is accuracy, which we have seen many times in past assignments.  Recall that the accuracy is given by
# 
# $$
# \mbox{accuracy} = \frac{\mbox{# correctly classified data points}}{\mbox{# total data points}}
# $$
# 
# To obtain the accuracy of our trained models using GraphLab Create, simply pass the option `metric='accuracy'` to the `evaluate` function. We compute the **accuracy** of our logistic regression model on the **test_data** as follows:

# In[18]:

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true=test_data['sentiment'], y_pred=model.predict(test_matrix))
print('Test accuracy: %s' % accuracy)


# ## Baseline: Majority class prediction
# 
# Recall from an earlier assignment that we used the **majority class classifier** as a baseline (i.e reference) model for a point of comparison with a more sophisticated classifier. The majority classifier model predicts the majority class for all data points. 
# 
# Typically, a good model should beat the majority class classifier. Since the majority class in this dataset is the positive class (i.e., there are more positive than negative reviews), the accuracy of the majority class classifier can be computed as follows:

# In[19]:

baseline = len(test_data[test_data['sentiment'] == 1])/len(test_data)
print("Baseline accuracy (majority class classifier): %s" % baseline)


# ** Quiz Question:** Using accuracy as the evaluation metric, was our **logistic regression model** better than the baseline (majority class classifier)?
# 
# Yes

# ## Confusion Matrix
# 
# The accuracy, while convenient, does not tell the whole story. For a fuller picture, we turn to the **confusion matrix**. In the case of binary classification, the confusion matrix is a 2-by-2 matrix laying out correct and incorrect predictions made in each label as follows:
# ```
#               +---------------------------------------------+
#               |                Predicted label              |
#               +----------------------+----------------------+
#               |          (+1)        |         (-1)         |
# +-------+-----+----------------------+----------------------+
# | True  |(+1) | # of true positives  | # of false negatives |
# | label +-----+----------------------+----------------------+
# |       |(-1) | # of false positives | # of true negatives  |
# +-------+-----+----------------------+----------------------+
# ```
# To print out the confusion matrix for a classifier, use `metric='confusion_matrix'`:

# In[21]:

from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_true=test_data['sentiment'],
                        y_pred=model.predict(test_matrix),
                        labels=model.classes_)
cmat


# In[24]:

print(' target_label | predicted_label | count ')
print('--------------+-----------------+-------')
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))


# **Quiz Question**: How many predicted values in the **test set** are **false positives**?

# In[25]:

1453


# ## Computing the cost of mistakes
# 
# 
# Put yourself in the shoes of a manufacturer that sells a baby product on Amazon.com and you want to monitor your product's reviews in order to respond to complaints.  Even a few negative reviews may generate a lot of bad publicity about the product. So you don't want to miss any reviews with negative sentiments --- you'd rather put up with false alarms about potentially negative reviews instead of missing negative reviews entirely. In other words, **false positives cost more than false negatives**. (It may be the other way around for other scenarios, but let's stick with the manufacturer's scenario for now.)
# 
# Suppose you know the costs involved in each kind of mistake: 
# 1. \$100 for each false positive.
# 2. \$1 for each false negative.
# 3. Correctly classified reviews incur no cost.
# 
# **Quiz Question**: Given the stipulation, what is the cost associated with the logistic regression classifier's performance on the **test set**?

# In[26]:

cost = 100*1453+1*804


# In[27]:

cost


# ## Precision and Recall

# You may not have exact dollar amounts for each kind of mistake. Instead, you may simply prefer to reduce the percentage of false positives to be less than, say, 3.5% of all positive predictions. This is where **precision** comes in:
# 
# $$
# [\text{precision}] = \frac{[\text{# positive data points with positive predicitions}]}{\text{[# all data points with positive predictions]}} = \frac{[\text{# true positives}]}{[\text{# true positives}] + [\text{# false positives}]}
# $$

# So to keep the percentage of false positives below 3.5% of positive predictions, we must raise the precision to 96.5% or higher. 
# 
# **First**, let us compute the precision of the logistic regression classifier on the **test_data**.

# In[28]:

from sklearn.metrics import precision_score
precision = precision_score(y_true=test_data['sentiment'], 
                            y_pred=model.predict(test_matrix))
print("Precision on test data: %s" % precision)


# **Quiz Question**: Out of all reviews in the **test set** that are predicted to be positive, what fraction of them are **false positives**? (Round to the second decimal place e.g. 0.25)

# In[29]:

1-precision


# **Quiz Question:** Based on what we learned in lecture, if we wanted to reduce this fraction of false positives to be below 3.5%, we would: (see the quiz)

# A complementary metric is **recall**, which measures the ratio between the number of true positives and that of (ground-truth) positive reviews:
# 
# $$
# [\text{recall}] = \frac{[\text{# positive data points with positive predicitions}]}{\text{[# all positive data points]}} = \frac{[\text{# true positives}]}{[\text{# true positives}] + [\text{# false negatives}]}
# $$
# 
# Let us compute the recall on the **test_data**.

# In[30]:

from sklearn.metrics import recall_score
recall = recall_score(y_true=test_data['sentiment'],
                      y_pred=model.predict(test_matrix))
print("Recall on test data: %s" % recall)


# **Quiz Question**: What fraction of the positive reviews in the **test_set** were correctly predicted as positive by the classifier?
# 
# 0.97
# 
# **Quiz Question**: What is the recall value for a classifier that predicts **+1** for all data points in the **test_data**?
# 
# 1

# # Precision-recall tradeoff
# 
# In this part, we will explore the trade-off between precision and recall discussed in the lecture.  We first examine what happens when we use a different threshold value for making class predictions.  We then explore a range of threshold values and plot the associated precision-recall curve.  
# 

# ## Varying the threshold
# 
# False positives are costly in our example, so we may want to be more conservative about making positive predictions. To achieve this, instead of thresholding class probabilities at 0.5, we can choose a higher threshold. 
# 
# Write a function called `apply_threshold` that accepts two things
# * `probabilities` (an SArray of probability values)
# * `threshold` (a float between 0 and 1).
# 
# The function should return an SArray, where each element is set to +1 or -1 depending whether the corresponding probability exceeds `threshold`.

# In[31]:

def apply_threshold(probabilities, threshold):
    prediction = pd.Series(probabilities).apply(lambda x: +1 if x > threshold else -1)
    return prediction


# Run prediction with `output_type='probability'` to get the list of probability values. Then use thresholds set at 0.5 (default) and 0.9 to make predictions from these probability values.

# In[32]:

probabilities = model.predict_proba(test_matrix)[:,1]
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)


# In[33]:

print("Number of positive predicted reviews (threshold = 0.5): %s" % (predictions_with_default_threshold == 1).sum())


# In[34]:

print("Number of positive predicted reviews (threshold = 0.9): %s" % (predictions_with_high_threshold == 1).sum())


# **Quiz Question**: What happens to the number of positive predicted reviews as the threshold increased from 0.5 to 0.9?
# 
# Decreased

# ## Exploring the associated precision and recall as the threshold varies

# By changing the probability threshold, it is possible to influence precision and recall. We can explore this as follows:

# In[35]:

precision_with_default_threshold = precision_score(y_true=test_data['sentiment'],
                                                   y_pred=predictions_with_default_threshold)
precision_with_high_threshold = precision_score(y_true=test_data['sentiment'],
                                                y_pred=predictions_with_high_threshold)


# In[37]:

recall_with_default_threshold = recall_score(y_true=test_data['sentiment'],
                                             y_pred=predictions_with_default_threshold)
recall_with_high_threshold = recall_score(y_true=test_data['sentiment'],
                                          y_pred=predictions_with_high_threshold)


# In[38]:

print("Precision (threshold = 0.5): %s" % precision_with_default_threshold)
print("Recall (threshold = 0.5)   : %s" % recall_with_default_threshold)


# In[39]:

print("Precision (threshold = 0.9): %s" % precision_with_high_threshold)
print("Recall (threshold = 0.9)   : %s" % recall_with_high_threshold)


# **Quiz Question (variant 1)**: Does the **precision** increase with a higher threshold?
# 
# True
# 
# **Quiz Question (variant 2)**: Does the **recall** increase with a higher threshold?
# 
# False

# ## Precision-recall curve
# 
# Now, we will explore various different values of tresholds, compute the precision and recall scores, and then plot the precision-recall curve.

# In[40]:

threshold_values = np.linspace(0.5, 1, num=100)
print(threshold_values)


# For each of the values of threshold, we compute the precision and recall scores.

# In[41]:

precision_all = []
recall_all = []

probabilities = model.predict_proba(test_matrix)[:,1]
for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    
    precision = precision_score(test_data['sentiment'], predictions)
    recall = recall_score(test_data['sentiment'], predictions)
    
    precision_all.append(precision)
    recall_all.append(recall)


# Now, let's plot the precision-recall curve to visualize the precision-recall tradeoff as we vary the threshold.

# In[42]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')


# **Quiz Question**: Among all the threshold values tried, what is the **smallest** threshold value that achieves a precision of 96.5% or better? Round your answer to 3 decimal places.

# In[43]:

new_list = []
for i in zip(threshold_values, precision_all):
    thre, preci = i
   
    if preci >= 0.965:
        new_list.append(thre) 
        
print(min(new_list))


# **Quiz Question**: Using `threshold` = 0.98, how many **false negatives** do we get on the **test_data**? (**Hint**: You may use the `graphlab.evaluation.confusion_matrix` function implemented in GraphLab Create.)

# In[45]:

cmat = confusion_matrix(y_true=test_data['sentiment'],
                        y_pred=apply_threshold(probabilities, 0.98),
                        labels=model.classes_)    # use the same order of class as the LR model.
print(' target_label | predicted_label | count ')
print('--------------+-----------------+-------')
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print('{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j]))


# In[46]:

8246


# This is the number of false negatives (i.e the number of reviews to look at when not needed) that we have to deal with using this classifier.

# # Evaluating specific search terms

# So far, we looked at the number of false positives for the **entire test set**. In this section, let's select reviews using a specific search term and optimize the precision on these reviews only. After all, a manufacturer would be interested in tuning the false positive rate just for their products (the reviews they want to read) rather than that of the entire set of products on Amazon.
# 
# ## Precision-Recall on all baby related items
# 
# From the **test set**, select all the reviews for all products with the word 'baby' in them.

# In[48]:

test_data['name'] = test_data['name'].astype('str')
baby_reviews =  test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]


# Now, let's predict the probability of classifying these reviews as positive:

# In[49]:

baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
probabilities = model.predict_proba(baby_matrix)[:,1]


# Let's plot the precision-recall curve for the **baby_reviews** dataset.
# 
# **First**, let's consider the following `threshold_values` ranging from 0.5 to 1:

# In[50]:

threshold_values = np.linspace(0.5, 1, num=100)


# **Second**, as we did above, let's compute precision and recall for each value in `threshold_values` on the **baby_reviews** dataset.  Complete the code block below.

# In[52]:

precision_all = []
recall_all = []

for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    
    precision = precision_score(baby_reviews['sentiment'], predictions)
    recall = recall_score(baby_reviews['sentiment'], predictions)
    
    precision_all.append(precision)
    recall_all.append(recall)
    
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')


# **Quiz Question**: Among all the threshold values tried, what is the **smallest** threshold value that achieves a precision of 96.5% or better for the reviews of data in **baby_reviews**? Round your answer to 3 decimal places.

# In[54]:

new_list = []
for i in zip(threshold_values, precision_all):
    thre, preci = i
   
    if preci >= 0.965:
        new_list.append(thre) 
        
print(min(new_list))


# **Quiz Question:** Is this threshold value smaller or larger than the threshold used for the entire dataset to achieve the same specified precision of 96.5%?
# 
# smaller 
# 
# **Finally**, let's plot the precision recall curve.
