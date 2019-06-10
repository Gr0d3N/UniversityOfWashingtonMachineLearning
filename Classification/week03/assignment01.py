
# coding: utf-8

# In[1]:

# imports
import pandas as pd


# # Loading the Lending Club dataset

# In[2]:

loans = pd.read_csv('lending-club-data.csv')


# # Exploring some features

# In[3]:

loans.columns


# # Exploring target column

# In[4]:

loans['safe_loans'] = np.where(loans.bad_loans == 0, 1, -1)
loans.drop('bad_loans', inplace=True, axis=1)
loans.head()


# In[5]:

# Exploring the distibution of the column safe_loans
print(loans['safe_loans'].value_counts()/len(loans))


# In[6]:

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]
loans.head()


# In[7]:

loans = pd.get_dummies(loans)


# In[8]:

loans.head()


# In[10]:

# split train test data

# getting the tarin indices
with open('module-5-assignment-1-train-idx.json') as f:
    train_idx = json.load(f)
    
# getting the test indices
with open('module-5-assignment-1-validation-idx.json') as f:
    validation_idx = json.load(f)
    
train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]


# # Build a decision tree classifier

# In[11]:

from sklearn.tree import DecisionTreeClassifier
import numpy as np


# In[13]:

decision_tree_model = DecisionTreeClassifier(max_depth=6)
X = train_data.drop('safe_loans', 1)
decision_tree_model.fit(X, train_data[target])


# In[14]:

small_model = DecisionTreeClassifier(max_depth=2)
small_model.fit(X, train_data[target])


# # Making predictions

# In[16]:

validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data


# In[18]:

sample_validation_data.safe_loans


# In[19]:

decision_tree_model.predict(sample_validation_data.drop('safe_loans',1))


# ## Quiz Question: What percentage of the predictions on sample_validation_data did decision_tree_model get correct?
# 50%

# # Explore probability predictions

# In[22]:

decision_tree_model.predict_proba(sample_validation_data.drop('safe_loans',1))


# ## Quiz Question: Which loan has the highest probability of being classified as a safe loan?
# The last 

# # Tricky predictions

# In[23]:

small_model.predict_proba(sample_validation_data.drop('safe_loans',1))


# ## Quiz Question: Notice that the probability preditions are the exact same for the 2nd and 3rd loans. Why would this happen?

# # Visualize the prediction on a tree

# ## Quiz Question: Based on the visualized tree, what prediction would you make for this data point (according to small_model)? (If you don't have Graphviz, you can answer this quiz question by executing the next part.)

# In[24]:

small_model.predict(sample_validation_data.drop('safe_loans',1))


# # Evaluate the accuracy of the decision tree model

# In[26]:

small_model.score(X, train_data[target])


# In[28]:

decision_tree_model.score(X, train_data[target])


# In[30]:

small_model.score(validation_data.drop('safe_loans', 1),
                  validation_data[target])


# In[31]:

decision_tree_model.score(validation_data.drop('safe_loans', 1),
                  validation_data[target])


# ## Quiz Question: What is the accuracy of decision_tree_model on the validation set, rounded to the nearest .01?
# 0.64

# # Evaluating accuracy of a complex decision tree model

# In[32]:

big_model = DecisionTreeClassifier(max_depth=10)
big_model.fit(X, train_data[target])


# In[33]:

big_model.score(X, train_data[target])


# In[34]:

big_model.score(validation_data.drop('safe_loans', 1),
                validation_data[target])


# ## How does the performance of big_model on the validation set compare to decision_tree_model on the validation set? Is this a sign of overfitting?
# less, yes

# # Quantifying the cost of mistakes

# In[35]:

validation_prediction = decision_tree_model.predict(validation_data.drop('safe_loans',1))
false_negative_counts = sum(validation_prediction < validation_data[target])
false_positive_counts = sum(validation_prediction > validation_data[target])
total_cost = 10000*false_negative_counts + 20000*false_positive_counts
total_cost


# ## Quiz Question: Let's assume that each mistake costs us money: a false negative costs \$10,000, while a false positive positive costs \$20,000. What is the total cost of mistakes made by decision_tree_model on validation_data?
# 50390000
