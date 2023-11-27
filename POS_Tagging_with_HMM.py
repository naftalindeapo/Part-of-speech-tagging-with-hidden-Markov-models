# ### Part-of-speech tagging with hidden Markov models
# ### *by Naftali N Indongo*

import re
import numpy as np
import pandas as pd
import seaborn as sns
from unidecode import unidecode
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


# ### (1.) Data
# In this assignment we will use the NCHLT Afrikaans Annotated Text Corpora downloaded from the Resource Catalogue of the 
# South African Centre for Digital Language Resources (SADiLAR) found at https://repo.sadilar.org).
# We will be using the excel files for training and test found the in the directory  2.POS Annotated.
import os
print("Current Working Directory:", os.getcwd())

# Reading in the datasets
af_test = pd.read_excel('Test_af.xls')
af_train = pd.read_excel('Train_af.xls')
print(af_test)

# Create a regular expression pattern to match website URLs
website_pattern = r'^https?://(?:www\.)?[\w-]+\.[\w.-]+[\w/]*$'

# Use the `str.contains` method to check if the Token column contains website URLs
website_instances = af_train[af_train['Token'].str.contains(website_pattern, case=False, na=False)]

# #### (2.) Data Normalization and Cleaning
#Delete the last row
af_test = af_test[:-1]
af_train = af_train[:-1]

def Begin_n_End_sentence(dataset):
    # Iterate over the rows of the DataFrame
    for i in range(len(dataset)):
        # Check if both Token and POS are NaN
        if pd.isna(dataset.at[i, 'Token']) and pd.isna(dataset.at[i, 'POS']):
            # Replace Token and POS with 'START' and '<s>'
            dataset.at[i, 'Token'] = 'START'
            dataset.at[i, 'POS'] = '<s>'

            # Check if there is a previous row
            if i > 0:
                # Replace Token and POS in the previous row with '</s>' and 'EOS'
                dataset.at[i-1, 'Token'] = 'END'
                dataset.at[i-1, 'POS'] = '</s>'
    
    # Return the updated dataset
    return dataset

A = Begin_n_End_sentence(af_train)

# Function to preprocess the data
def preprocess_data(data):
    
    #Modelling the beginning of the first setentence by inserting a new row with Token = '<s>' and POS = 'BOS'
    first_row = pd.DataFrame({'Token': ['START'], 'POS': ['<s>']})

    # Concatenate the new row with the original DataFrame
    data = pd.concat([first_row, data]).reset_index(drop=True)
    
    #Model the beginning and end of sentence
    data = Begin_n_End_sentence(data)
    
    # Convert the 'Token' column to lowercase
    data['Token'] = data['Token'].apply(lambda x: x.lower() if x not in ['START', 'END'] else x)


    # Remove punctuation from the Token column, excluding '<s>' and '</s>'
    data['Token'] = data['Token'].apply(lambda x: re.sub(r'[^\w\s]', '', x) \
                                          if isinstance(x, str) and x != '</s>' and x!='<s>' else x)
    # Convert diacritics to closest ASCII representation in the Token column
    data['Token'] = data['Token'].apply(lambda x: unidecode(x) if isinstance(x, str) else x)
    # Drop the rows where the value in the 'Token' column is a space (' ')
    data = data[data['Token'] != '']
    
    #Modelling the end of the last setentence by inserting a new row with Token = '</s>' and POS = 'EOS'
    last_row = pd.DataFrame({'Token': ['END'], 'POS': ['</s>']})

    # Concatenate the new row with the original DataFrame
    data = pd.concat([data, last_row]).reset_index(drop=True)
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Reset the row indices
    data = data.reset_index(drop=True)
    return data

af_test = preprocess_data(af_test)
af_train = preprocess_data(af_train)


# Check for missing values in the entire DataFrame
missing_values = af_train.isnull().sum()

# Print the number of missing values for each column
print(missing_values)


af_test.tail()

# splitting into training and validation
Train_set = af_train.iloc[:39064]
Val_set = af_train.iloc[39064:]

# create list of training, validation and test tagged words
Train_tagged_words = [(row["Token"], row["POS"]) for _, row in Train_set.iterrows()]
Val_tagged_words = [(row["Token"], row["POS"]) for _, row in Val_set.iterrows()]
Test_tagged_words = [(row["Token"], row["POS"]) for _, row in af_test.iterrows()]

# Checking how many unique tags are present in training data
Unique_tags = [tag for word,tag in Train_tagged_words]
 
# check total words in the training vocabulary
vocab = {word for word,tag in Train_tagged_words}


# #### 3. Converting each dataset into a list of lists of tupples (word, tag) and splitting the training set into training and validation

# Below we will convert the training and test sets into lists of lists of tupples (word, tag), where each inner list corresponds to a sentence.
#  We will also split the training set into training and validation.

def create_sentence_boundaries(dataset):
    sentence_boundaries = []  # List to store the indices where sentences end

    # Iterate over the rows of the dataset
    for i in range(len(dataset)):
        # Check if the Token is 'END'
        if dataset.at[i, 'Token'] == 'END':
            sentence_boundaries.append(i+1)  # Append the index of the next row

    # Return the list of sentence boundaries
    return sentence_boundaries

def convert_dataset(dataset, sentence_boundaries):
    sentences = []  # List to store the converted dataset

    start_index = 0  # Start index of the current sentence

    # Iterate over the sentence boundaries
    for end_index in sentence_boundaries:
        sentence = []  # List to store the tuples of (token, tag) for the current sentence

        # Iterate over the rows within the current sentence boundaries
        for i in range(start_index, end_index):
            token = dataset.at[i, 'Token']
            tag = dataset.at[i, 'POS']
            sentence.append((token, tag))

        sentences.append(sentence)  # Append the sentence to the list of sentences
        start_index = end_index  # Update the start index for the next sentence

    return sentences

Test_indices = create_sentence_boundaries(af_test)
Train_indices = create_sentence_boundaries(af_train)
af_test_list = convert_dataset(af_test, Test_indices)
af_train_list = convert_dataset(af_train, Train_indices)


# Splitting into training and validation, and converting to arrays.
Train_data = af_train_list[:2090]
Val_data = af_train_list[2090:]
Test_data = af_test_list


# ### Part 1: Developing a basic HMM tagger

# #### (a) Dataset splitting
# List of tupples (token, tag) for the datasets
Train_word_tag = [ tup for sent in Train_data for tup in sent ]
Val_word_tag = [ tup for sent in Val_data for tup in sent]
Test_word_words = [ tup for sent in Test_data for tup in sent]

#Getting the tags and tagged words
Training_tags = [tag for word,tag in Train_word_tag]
Train_tagged_words = [word for word,tag in Train_word_tag]

# Checking how many unique tags are present in training data
Unique_tags = {tag for word,tag in Train_word_tag}
print('There are {} training tags.'.format(len(Unique_tags)))
 
# check the unique word tokens in the training vocabulary
Training_vocab = {word for word,tag in Train_word_tag}
print('There are {} unique word tokens in the training set'.format(len(Training_vocab)))


# #### (b) Model splitting
# function for getting the n-grams
def ngrams(sentence, n):
    Ngrams = []
    for i in range(len(sentence)):
        Ngrams.append(tuple(sentence[i: i + n]))
    return Ngrams

# function for getting bigram count
def bigram_counts(tags):
    bigram_cnt = {}
    for i_tag_bigram in ngrams(tags, 2):
        if i_tag_bigram in bigram_cnt:
            bigram_cnt[i_tag_bigram] += 1
        else:
            bigram_cnt[i_tag_bigram] = 1
    return bigram_cnt

#function for getting unigram count
def unigram_counts(tags):
    unigram_cnt = {}
    for tag in tags:
        if tag in unigram_cnt:
            unigram_cnt[tag] += 1
        else:
            unigram_cnt[tag] = 1
    return unigram_cnt


#function for getting tagged word count
def tag_word_counts(tagged_words):
    tag_count = defaultdict(lambda: 0)
    tag_word_count = Counter()
    for word, tag in tagged_words:
        tag_count[tag] += 1
        if (word, tag) in tag_word_count:
            tag_word_count[(word, tag)] += 1
        else:
            tag_word_count[(word, tag)] = 1
    return tag_count, tag_word_count

# Estimating the transition probabilities
def transition_probabilty(tags, bigram_cnt, unigram_cnt):
    transition_probabilities = defaultdict(lambda: 0)
    bigrams = ngrams(tags, 2)
    for bigram in bigrams:
        transition_probabilities[bigram] = bigram_cnt[bigram] / unigram_cnt[bigram[0]]
    return transition_probabilities

# Estimate the emmission probabilities 
def emmission_probabilty(tag_word_count,tagged_words, tag_count):
    emmission_probabilities = defaultdict(lambda: 0)
    for word, tag in tagged_words:
        emmission_probabilities[(word, tag)] = tag_word_count[(word, tag)] / tag_count[tag]
    return emmission_probabilities

######### Estimating the transition probabilities############
# 1. Calculate the bigram counts C(tag_t,tag_{t-1})
Bigram_counts = bigram_counts(Training_tags)

# 2. Calculate the unigram counts C(tag_{t-1})
Unigram_counts = unigram_counts(Training_tags)

# 3. Compute the transition probabilities
Transition_probs = transition_probabilty(Training_tags, Bigram_counts, Unigram_counts)

######### Estimating the emission probabilities############
# 1. Calculate the tag_counts and tag_word counts C(tag_t) and C(word_t,tag_t})
tag_counts, tag_word_counts = tag_word_counts(Train_word_tag)
print(len(tag_word_counts))
print(len(tag_counts))

# 2. Compute the transition probabilities
Emmission_Probs = emmission_probabilty(tag_word_counts,Train_word_tag, tag_counts)


# #### (c) The Viterbi algorithm for tagging new sentences

def viterbi_decoder(sentence, transition_probs, emission_probs):
    observable = [word for word, tag in sentence]
    in_states = [tag for word, tag in sentence]
    states = in_states
    K = len(states)
    deltas = {}
    phi_s = {}
    
    unk_prob = 0.00001
    start_tag = '<s>'
    end_tag = '</s>'
    
    # Initialization
    for j, state in enumerate(states):
        deltas[state, 1] = emission_probs.get((observable[1], state), unk_prob) * transition_probs.get((start_tag, state), unk_prob)

    # Recursion
    for t in range(2, len(observable)):
        obs = observable[t]
        for j, state in enumerate(states):
            max_prob = 0.0
            max_index = 0
            for i, prev_state in enumerate(states):
                prob = deltas[prev_state, t-1] * transition_probs.get((prev_state, state), unk_prob) * emission_probs.get((obs, state), unk_prob)
                if prob > max_prob:
                    max_prob = prob
                    max_index = i
            deltas[state, t] = max_prob
            phi_s[state, t] = max_index

    # Termination
    max_prob = 0.0
    max_index = 0
    for j, state in enumerate(states):
        prob = transition_probs.get((state, end_tag), unk_prob) * deltas[state, len(observable) - 1]
        if prob > max_prob:
            max_prob = prob
            max_index = j

    # Backtracking
    best_path = []
    best_path.append(start_tag)  # Add the start tag at the beginning
    best_path.append(states[max_index])
    for t in range(len(observable) - 1, 1, -1):
        max_index = phi_s[states[max_index], t]
        best_path.insert(1, states[max_index])  # Insert the tag at the second position to maintain the order
    
    return best_path


#Test
Test_results = viterbi_decoder(Val_data[0], Transition_probs, Emmission_Probs)
# Some examples:
Sentence = [('START', '<s>'), ('verstrek', 'VTHOG'), ('die', 'LB'), ('in', 'SVS'), ('die', 'LB'), ('afdeling', 'NSE'),\
     ('manufacturer', 'RV'), ('for', 'RV'), ('agoa', 'RK'), ('particulars', 'RV'), ('END', '</s>')]

# Get the actual tags and predicted tags
Actual_tags = [tag for word,tag in Sentence]
Words = [word for word, tag in Sentence]

# Get the predicted tags
Predicted_tags = viterbi_decoder(Sentence, Transition_probs, Emmission_Probs)

Results_table = pd.DataFrame({'Token': Words, 'Actual_tags': Actual_tags, 'Predicted_tags': Predicted_tags})
print(Results_table)


# #### (d.) Model Evaluation
#function for calculating the accuracy on a given set
def calculate_accuracy(given_set, transition_probs, emission_probs):
    correct_tags = 0
    total_tags = 0
    
    for sentence in given_set:
        predicted_tags = viterbi_decoder(sentence, transition_probs, emission_probs)
        actual_tags = [tag for _, tag in sentence]
        
        for predicted_tag, actual_tag in zip(predicted_tags, actual_tags):
            if predicted_tag == actual_tag:
                correct_tags += 1
            total_tags += 1
    
    accuracy = correct_tags / total_tags
    return accuracy

# Assuming you have a validation_set containing sentences in the same format as your dataset
accuracy = calculate_accuracy(Test_data, Transition_probs, Emmission_Probs)
print("Accuracy:", accuracy)

############# Calculating the accuracy of our HMM tagger on the validation and test set #########
Val_accuracy =  calculate_accuracy(Val_data, Transition_probs, Emmission_Probs)
Test_accuracy = calculate_accuracy(Test_data, Transition_probs, Emmission_Probs)

print(f'The accuracy on the validation set is {Val_accuracy:.4%}')
print(f'The accuracy on the test set is {Test_accuracy:.4%}')