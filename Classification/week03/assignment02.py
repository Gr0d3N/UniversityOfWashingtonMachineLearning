
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np


# # Loading the data

# In[4]:

loans = pd.read_csv('lending-club-data.csv')
loans.head()


# In[5]:

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis=1)


# In[6]:

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'


# In[7]:

loans = loans[features + [target]]
loans.head()


# In[8]:

loans = pd.get_dummies(loans)


# In[9]:

loans.head()


# In[10]:

# split train test data

# getting the tarin indices
with open('module-5-assignment-2-train-idx.json') as f:
    train_idx = json.load(f)
    
# getting the test indices
with open('module-5-assignment-2-test-idx.json') as f:
    test_idx = json.load(f)
    
train_data = loans.iloc[train_idx]
test_data = loans.iloc[test_idx]


# # Function to count number of mistakes while predicting majority class

# In[11]:

safe_loans_count = sum(loans[target] == 1)
risky_loans_count = sum(loans[target] == -1)
print(safe_loans_count)
print(risky_loans_count)
print(min(safe_loans_count, risky_loans_count))


# In[12]:

def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    
    # Count the number of 1's (safe loans)
    safe_loans = sum(labels_in_node == 1)
    
    # Count the number of -1's (risky loans)
    risky_loans = sum(labels_in_node == -1)
    
    # Return the number of mistakes that the majority classifier makes
    return min(safe_loans, risky_loans)


# In[13]:

# Test case 1
example_labels = np.array([-1, -1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print('Test passed!')
else:
    print('Test 1 failed... try again!')

# Test case 2
example_labels = np.array([-1, -1, 1, 1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print('Test passed!')
else:
    print('Test 3 failed... try again!')
    
# Test case 3
example_labels = np.array([-1, -1, -1, -1, -1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print('Test passed!')
else:
    print('Test 3 failed... try again!')


# # Function to pick best feature to split on

# In[14]:

def best_splitting_feature(data, features, target):
    
    target_values = data[target]
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        ## YOUR CODE HERE
        right_split = data[data[feature] == 1] 
            
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        # YOUR CODE HERE
        left_mistakes = intermediate_node_num_mistakes(left_split[target])            

        # Calculate the number of misclassified examples in the right split.
        ## YOUR CODE HERE
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        ## YOUR CODE HERE
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        ## YOUR CODE HERE
        if error < best_error:
            best_error = error
            best_feature = feature
    
    return best_feature # Return the best feature we found


# # Building the tree

# In[15]:

def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True    }   ## YOUR CODE HERE 
   
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = +1         ## YOUR CODE HERE
    else:
        leaf['prediction'] = -1         ## YOUR CODE HERE        

    # Return the leaf node
    return leaf 


# In[26]:

def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(target_values) == 0:  ## YOUR CODE HERE
        print("Stopping condition 1 reached.")    
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:   ## YOUR CODE HERE
        print("Stopping condition 2 reached.")
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  ## YOUR CODE HERE
        print("Reached maximum depth. Stopping for now.")
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    ## YOUR CODE HERE
    splitting_feature = best_splitting_feature(data, features, target)

    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]       ## YOUR CODE HERE
    remaining_features.remove(splitting_feature)
    print("Split on feature %s. (%s, %s)" % (                      splitting_feature, len(left_split), len(right_split)))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(right_split[target])

        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    ## YOUR CODE HERE
    right_tree = decision_tree_create(
        right_split, remaining_features, target, current_depth + 1, max_depth)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


# In[27]:

x = train_data.drop('safe_loans', 1)
features_new = [col for col in x.columns]
my_decision_tree = decision_tree_create(train_data, features_new, target, current_depth=0, max_depth=6)


# # Making predictions

# In[37]:

def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)


# In[38]:

for a in test_data[0:1].to_dict(orient = 'records'):
    print(a)


# In[39]:

print('Predicted class: %s ' % classify(my_decision_tree, a))


# In[40]:

classify(my_decision_tree, a, annotate=True)


# ## Quiz question: What was the feature that my_decision_tree first split on while making the prediction for test_data[0]?
# term_ 36 months

# ## Quiz question: What was the first feature that lead to a right split of test_data[0]?
# grade_D

# ## Quiz question: What was the last feature split on before reaching a leaf node for test_data[0]?
# grade_D

# # Evaluating your decision tree

# In[44]:

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    data['prediction'] = [classify(tree,a) for a in data.to_dict(orient = 'records')]
    
    # Once you've made the predictions, calculate the classification error and return it
    ## YOUR CODE HERE
    classification_error = round(float(sum(data['prediction'] != data['safe_loans']))/len(data),2)
    return classification_error


# In[45]:

evaluate_classification_error(my_decision_tree, test_data)


# ## Quiz Question: Rounded to 2nd decimal point, what is the classification error of my_decision_tree on the test_data?
# 0.38

# # Printing out a decision stump

# In[46]:

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print("(leaf, label: %s)" % tree['prediction'])
        return None
    split_feature, split_value = split_name.split('_')
    print('                       %s' % name)
    print('         |---------------|----------------|')
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('  [{0} == 0]               [{0} == 1]    '.format(split_name))
    print('         |                                |')
    print('         |                                |')
    print('         |                                |')
    print('    (%s)                         (%s)'         % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree')))


# In[47]:

print_stump(my_decision_tree, name = 'root')


# ## Quiz Question: What is the feature that is used for the split at the root node?
# term_ 36 months

# # Exploring the intermediate left subtree

# In[48]:

print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])


# In[49]:

print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])


# In[50]:

print_stump(my_decision_tree['right'], my_decision_tree['splitting_feature'])


# In[51]:

print_stump(my_decision_tree['right']['right'], my_decision_tree['right']['splitting_feature'])


# ## Quiz question: What is the path of the first 3 feature splits considered along the left-most branch of my_decision_tree?

# ## Quiz question: What is the path of the first 3 feature splits considered along the right-most branch of my_decision_tree?

# In[ ]:



