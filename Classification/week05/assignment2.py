
# coding: utf-8

# #Boosting a decision stump

# In this homework you will implement your own boosting module.
# 
# Brace yourselves! This is going to be a fun and challenging assignment.
# 
# Use SFrames to do some feature engineering.
# Train a boosted ensemble of decision-trees (gradient boosted trees) on the lending club dataset.
# Predict whether a loan will default along with prediction probabilities (on a validation set).
# Evaluate the trained model and compare it with a baseline.
# Find the most positive and negative loans using the learned model.
# Explore how the number of trees influences classification performance.
# 

# #Load the landing club dataset

# In[1]:

import pandas as pd
import numpy as np
loans = pd.read_csv('/Users/April/Downloads/lending-club-data.csv')


# In[2]:

# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis = 1)


# we select four categorical features:
# 
# grade of the loan
# the length of the loan term
# the home ownership status: own, mortgage, rent
# number of years of employment.

# In[3]:

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]

target = 'safe_loans'
loans = loans[features + [target]]


# In[4]:

loans = pd.get_dummies(loans)


# In[5]:

import json
with open('/Users/April/Desktop/datasci_course_materials-master/assignment1/train index.json', 'r') as f: # Reads the list of most frequent words
    train_idx = json.load(f)
with open('/Users/April/Desktop/datasci_course_materials-master/assignment1/validation index.json', 'r') as f1: # Reads the list of most frequent words
    validation_idx = json.load(f1)


# In[6]:

train_data = loans.iloc[train_idx].reset_index()
validation_data = loans.iloc[validation_idx].reset_index()


# In[7]:

train_data = train_data.drop('index', 1)
validation_data = validation_data.drop('index',1)


# #Weighted decision trees

# Let's modify our decision tree code from Module 5 to support weighting of individual data points.

# Weighted error definition
# Consider a model with $N$ data points with:
# Predictions $\hat{y}_1 ... \hat{y}_n$
# Target $y_1 ... y_n$
# Data point weights $\alpha_1 ... \alpha_n$.
# Then the weighted error is defined by: $$
# \mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \frac{\sum_{i=1}^{n} \alpha_i \times 1[y_i \neq \hat{y_i}]}{\sum_{i=1}^{n} \alpha_i}
# $$ where $1[y_i \neq \hat{y_i}]$ is an indicator function that is set to $1$ if $y_i \neq \hat{y_i}$.
# 

# Write a function to compute weight of mistakes
# Write a function that calculates the weight of mistakes for making the "weighted-majority" predictions for a dataset. The function accepts two inputs:
# labels_in_node: Targets $y_1 ... y_n$
# data_weights: Data point weights $\alpha_1 ... \alpha_n$
# We are interested in computing the (total) weight of mistakes, i.e. $$
# \mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \sum_{i=1}^{n} \alpha_i \times 1[y_i \neq \hat{y_i}].
# $$ This quantity is analogous to the number of mistakes, except that each mistake now carries different weight. It is related to the weighted error in the following way: $$
# \mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \frac{\mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}})}{\sum_{i=1}^{n} \alpha_i}
# $$
# The function intermediate_node_weighted_mistakes should first compute two weights:
# $\mathrm{WM}_{-1}$: weight of mistakes when all predictions are $\hat{y}_i = -1$ i.e $\mathrm{WM}(\mathbf{\alpha}, \mathbf{-1}$)
# $\mathrm{WM}_{+1}$: weight of mistakes when all predictions are $\hat{y}_i = +1$ i.e $\mbox{WM}(\mathbf{\alpha}, \mathbf{+1}$)
# where $\mathbf{-1}$ and $\mathbf{+1}$ are vectors where all values are -1 and +1 respectively.
# After computing $\mathrm{WM}_{-1}$ and $\mathrm{WM}_{+1}$, the function intermediate_node_weighted_mistakes should return the lower of the two weights of mistakes, along with the class associated with that weight. 

# In[8]:

def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    # Sum the weights of all entries with label +1
    total_weight_positive = sum(data_weights[labels_in_node == +1])
    
    # Weight of mistakes for predicting all -1's is equal to the sum above
    ### YOUR CODE HERE
    weighted_mistakes_all_negative = total_weight_positive
    
    # Sum the weights of all entries with label -1
    ### YOUR CODE HERE
    total_weight_negative = sum(data_weights[labels_in_node == -1])
    
    # Weight of mistakes for predicting all +1's is equal to the sum above
    ### YOUR CODE HERE
    weighted_mistakes_all_positive = total_weight_negative
    
    # Return the tuple (weight, class_label) representing the lower of the two weights
    #    class_label should be an integer of value +1 or -1.
    # If the two weights are identical, return (weighted_mistakes_all_positive,+1)
    ### YOUR CODE HERE
    if weighted_mistakes_all_negative < weighted_mistakes_all_positive:
        return (weighted_mistakes_all_negative, -1)
    else: 
        return (weighted_mistakes_all_positive, +1)
    


# Recall that the classification error is defined as follows:

# Quiz Question: If we set the weights α=1 for all data points, how is the weight of mistakes WM(α,ŷ) related to the classification error

# We continue modifying our decision tree code from the earlier assignment to incorporate weighting of individual data points. The next step is to pick the best feature to split on.
# The best_splitting_feature function is similar to the one from the earlier assignment with two minor modifications:
# The function best_splitting_feature should now accept an extra parameter data_weights to take account of weights of data points.
# Instead of computing the number of mistakes in the left and right side of the split, we compute the weight of mistakes for both sides, add up the two weights, and divide it by the total weight of the data.
# Complete the following function. Comments starting with DIFFERENT HERE mark the sections where the weighted version differs from the original implementation.

# #Function to pick best feature to split on

# We continue modifying our decision tree code from the earlier assignment to incorporate weighting of individual data points. The next step is to pick the best feature to split on.
# 
# The best_splitting_feature function is similar to the one from the earlier assignment with two minor modifications:
# 
# The function best_splitting_feature should now accept an extra parameter data_weights to take account of weights of data points.
# Instead of computing the number of mistakes in the left and right side of the split, we compute the weight of mistakes for both sides, add up the two weights, and divide it by the total weight of the data.

# In[9]:

# If the data is identical in each feature, this function should return None

def best_splitting_feature(data, features, target, data_weights):
    
    # These variables will keep track of the best feature and the corresponding error
    best_feature = None
    best_error = float('+inf') 
    num_points = float(len(data))

    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        # The right split will have all data points where the feature value is 1
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]
        
        # Apply the same filtering to data_weights to create left_data_weights, right_data_weights
        ## YOUR CODE HERE
        left_data_weights = data_weights[left_split.index]
        right_data_weights = data_weights[right_split.index]
                    
        # DIFFERENT HERE
        # Calculate the weight of mistakes for left and right sides
        ## YOUR CODE HERE
        left_weighted_mistakes, left_class = intermediate_node_weighted_mistakes(left_split[target], left_data_weights)
        right_weighted_mistakes, right_class = intermediate_node_weighted_mistakes(right_split[target], right_data_weights)
        
        # DIFFERENT HERE
        # Compute weighted error by computing
        #  ( [weight of mistakes (left)] + [weight of mistakes (right)] ) / [total weight of all data points]
        ## YOUR CODE HERE
        error = left_weighted_mistakes + right_weighted_mistakes/sum(data_weights)
        
        # If this is the best error we have found so far, store the feature and the error
        if error < best_error:
            best_feature = feature
            best_error = error
    
    # Return the best feature we found
    return best_feature


# Very Optional. Relationship between weighted error and weight of mistakes
# By definition, the weighted error is the weight of mistakes divided by the weight of all data points, so $$
# \mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \frac{\sum_{i=1}^{n} \alpha_i \times 1[y_i \neq \hat{y_i}]}{\sum_{i=1}^{n} \alpha_i} = \frac{\mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}})}{\sum_{i=1}^{n} \alpha_i}.
# $$
# In the code above, we obtain $\mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}})$ from the two weights of mistakes from both sides, $\mathrm{WM}(\mathbf{\alpha}_{\mathrm{left}}, \mathbf{\hat{y}}_{\mathrm{left}})$ and $\mathrm{WM}(\mathbf{\alpha}_{\mathrm{right}}, \mathbf{\hat{y}}_{\mathrm{right}})$

# First, notice that the overall weight of mistakes $\mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}})$ can be broken into two weights of mistakes over either side of the split: $$ \mathrm{WM}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \sum_{i=1}^{n} \alpha_i \times 1[y_i \neq \hat{y_i}] = \sum_{\mathrm{left}} \alpha_i \times 1[y_i \neq \hat{y_i}]
# \sum_{\mathrm{right}} \alpha_i \times 1[y_i \neq \hat{y_i}]\ = \mathrm{WM}(\mathbf{\alpha}{\mathrm{left}}, \mathbf{\hat{y}}{\mathrm{left}}) + \mathrm{WM}(\mathbf{\alpha}{\mathrm{right}}, \mathbf{\hat{y}}{\mathrm{right}}) $$
# We then divide through by the total weight of all data points to obtain $\mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}})$:
# $$ \mathrm{E}(\mathbf{\alpha}, \mathbf{\hat{y}}) = \frac{\mathrm{WM}(\mathbf{\alpha}<em>{\mathrm{left}}, \mathbf{\hat{y}}</em>{\mathrm{left}}) + \mathrm{WM}(\mathbf{\alpha}<em>{\mathrm{right}}, \mathbf{\hat{y}}</em>{\mathrm{right}})}{\sum_{i=1}^{n} \alpha_i} $$

# ##building the tree

# With the above functions implemented correctly, we are now ready to build our decision tree. Recall from the previous assignments that each node in the decision tree is represented as a dictionary which contains the following keys:

# { 
#    'is_leaf'            : True/False.
#    'prediction'         : Prediction at the leaf node.
#    'left'               : (dictionary corresponding to the left tree).
#    'right'              : (dictionary corresponding to the right tree).
#    'features_remaining' : List of features that are posible splits.
# }

# Let us start with a function that creates a leaf node given a set of target values. 

# In[10]:

def create_leaf(target_values, data_weights):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'is_leaf': True}
    
    # Computed weight of mistakes.
    # Store the predicted class (1 or -1) in leaf['prediction']
    weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class
    
    return leaf


# Now write a function that learns a weighted decision tree recursively and implements 3 stopping conditions:
# 
# All data points in a node are from the same class.
# No more features to split on.
# Stop growing the tree when the tree depth reaches max_depth.
# 

# In[11]:

def weighted_decision_tree_create(data, features, target, data_weights, current_depth = 1, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    # Stopping condition 1. Error is 0.
    if intermediate_node_weighted_mistakes(target_values, data_weights)[0] <= 1e-15:
        print "Stopping condition 1 reached."                
        return create_leaf(target_values, data_weights)
    
    # Stopping condition 2. No more features.
    if remaining_features == []:
        print "Stopping condition 2 reached."                
        return create_leaf(target_values, data_weights)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth > max_depth:
        print "Reached maximum depth. Stopping for now."
        return create_leaf(target_values, data_weights)
    
    # If all the datapoints are the same, splitting_feature will be None. Create a leaf
    splitting_feature = best_splitting_feature(data, features, target, data_weights)
    remaining_features.remove(splitting_feature)
        
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]
    
    print "Split on feature %s. (%s, %s)" % (              splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target], data_weights)
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target], data_weights)
    
    # Repeat (recurse) on left and right subtrees
    left_tree = weighted_decision_tree_create(
        left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)
    right_tree = weighted_decision_tree_create(
        right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


# Finally, write a recursive function to count the nodes in your tree. 

# In[12]:

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


# #Making predictions with a weighted decision tree

# To make a single prediction, we must start at the root and traverse down the decision tree in recursive fashion. 

# In[13]:

def classify(tree, x, annotate = False):   
    # If the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # Split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)


# #Evaluating the tree
Create a function called evaluate_classification_error. It takes in as input:

tree (as described above)
data (an data frame)
The function does not change because of adding data point weights. 
# In[14]:

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    data['prediction'] = [classify(tree,a) for a in data.to_dict(orient = 'records')]
    
    # Once you've made the predictions, calculate the classification error
    return (data['prediction'] != data[target]).sum() / float(len(data))

Example: Training a weighted decision treeTo build intuition on how weighted data points affect the tree being built, consider the following:

Suppose we only care about making good predictions for the first 10 and last 10 items in train_data, we assign weights:

1 to the last 10 items
1 to the first 10 items
and 0 to the rest.
Let us fit a weighted decision tree with max_depth = 2. Then compute the classification error on the subset_20, i.e. the subset of data points whose weight is 1 (namely the first and last 10 data points).
# In[15]:

# Assign weights
example_data_weights = np.ones(10*1).tolist() + [0.]*(len(train_data) - 20) + np.ones(1*10).tolist()
example_data_weights = np.array(example_data_weights)


# In[16]:

example_data_weights


# In[17]:

features = [i for i in train_data.columns]


# In[18]:

features.remove('safe_loans')


# In[19]:

example_data_weights


# In[20]:

# Train a weighted decision tree model.
small_data_decision_tree_subset_20 = weighted_decision_tree_create(train_data, features, target, 
                                                                   example_data_weights,  max_depth=2)

Now, we will compute the classification error on the subset_20, i.e. the subset of data points whose weight is 1 (namely the first and last 10 data points).

evaluate_classification_error(small_data_decision_tree_subset_20, train_data)

# In[21]:

subset_20 = train_data.head(10).append(train_data.tail(10))


# In[22]:

small_data_decision_tree_subset_20['splitting_feature']


# In[23]:

evaluate_classification_error(small_data_decision_tree_subset_20, subset_20)

The model small_data_decision_tree_subset_20 performs a lot better on subset_20 than on train_data.

So, what does this mean?

The points with higher weights are the ones that are more important during the training process of the weighted decision tree.
The points with zero weights are basically ignored during training.
Quiz Question: Will you get the same model as small_data_decision_tree_subset_20 if you trained a decision tree with only the 20 data points with non-zero weights from the set of points in subset_20?
# In[24]:

evaluate_classification_error(small_data_decision_tree_subset_20, train_data)


# #Implementing your own Adaboost (on decision stumps)

# Now that we have a weighted decision tree working, it takes only a bit of work to implement Adaboost. For the sake of simplicity, let us stick with decision tree stumps by training trees with max_depth=1.
# Recall from the lecture the procedure for Adaboost:
# 1. Start with unweighted data with $\alpha_j = 1$
# 2. For t = 1,...T:
# Learn $f_t(x)$ with data weights $\alpha_j$
# Compute coefficient $\hat{w}_t$: $$\hat{w}_t = \frac{1}{2}\ln{\left(\frac{1- \mbox{E}(\mathbf{\alpha}, \mathbf{\hat{y}})}{\mbox{E}(\mathbf{\alpha}, \mathbf{\hat{y}})}\right)}$$
# Re-compute weights $\alpha_j$: $$\alpha_j \gets \begin{cases}
#  \alpha_j \exp{(-\hat{w}_t)} &amp; \text{ if }f_t(x_j) = y_j\\
#  \alpha_j \exp{(\hat{w}_t)} &amp; \text{ if }f_t(x_j) \neq y_j
#  \end{cases}$$
# Normalize weights $\alpha_j$: $$\alpha_j \gets \frac{\alpha_j}{\sum_{i=1}^{N}{\alpha_i}} $$
# Complete the skeleton for the following code to implement adaboost_with_tree_stumps. 

# In[25]:

from math import log
from math import exp

def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
    # start with unweighted data
    alpha = np.ones(len(data)*1)
    weights = []
    tree_stumps = []
    target_values = data[target]
    
    for t in xrange(num_tree_stumps):
        print '====================================================='
        print 'Adaboost Iteration %d' % t
        print '====================================================='        
        # Learn a weighted decision tree stump. Use max_depth=1
        tree_stump = weighted_decision_tree_create(data, features, target, data_weights=alpha, max_depth=1)
        tree_stumps.append(tree_stump)
        
        # Make predictions
        data['prediction'] = [classify(tree_stump,a) for a in data.to_dict(orient = 'records')]
        
        # Produce a Boolean array indicating whether
        # each data point was correctly classified
        is_correct = data['prediction'] == target_values
        is_wrong   = data['prediction'] != target_values
        
        # Compute weighted error
        # YOUR CODE HERE
        weighted_error = round(float(sum(a for i, a in enumerate(alpha) if is_wrong[i]))/sum(alpha),2)
        
        # Compute model coefficient using weighted error
        # YOUR CODE HERE
        weight = log((1-weighted_error)/weighted_error)/2
        weights.append(weight)
        
        # Adjust weights on data point
        adjustment = is_correct.apply(lambda is_correct : exp(-weight) if is_correct else exp(weight))
        
        # Scale alpha by multiplying by adjustment
        # Then normalize data points weights
        ## YOUR CODE HERE 
        sum_alpha = sum(alpha)
        alpha = alpha*adjustment/sum_alpha
    
    return weights, tree_stumps


# In[26]:

stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=2)


# In[27]:

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('_')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)'         % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))


# In[28]:

print_stump(tree_stumps[0])


# In[29]:

print_stump(tree_stumps[1])


# #Training a boosted ensemble of 10 stumps

# Recall from the lecture that in order to make predictions, we use the following formula: $$
# \hat{y} = sign\left(\sum_{t=1}^T \hat{w}_t f_t(x)\right)
# $$
# We need to do the following things:
# Compute the predictions $f_t(x)$ using the $t$-th decision tree
# Compute $\hat{w}_t f_t(x)$ by multiplying the stump_weights with the predictions $f_t(x)$ from the decision trees
# Sum the weighted predictions over each stump in the ensemble.
# Complete the following skeleton for making predictions:

# In[34]:

def predict_adaboost(stump_weights, tree_stumps, data):
    scores = np.zeros(len(data)*1)
    
    for i, tree_stump in enumerate(tree_stumps):
        data['prediction'] = [classify(tree_stump,a) for a in data.to_dict(orient = 'records')]
        
        # Accumulate predictions on scaores array
        # YOUR CODE HERE
        scores += stump_weights[i] * data['prediction']
        data['scores'] = scores
        
    return data['scores'].apply(lambda score : +1 if score > 0 else -1)


# In[35]:

predictions = predict_adaboost(stump_weights, tree_stumps, validation_data)


# In[38]:

accuracy = float(sum(validation_data[target] == predictions))/len(validation_data)


# In[39]:

accuracy


# In[40]:

stump_weights


# In[ ]:



