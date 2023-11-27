# Part-of-speech-tagging-with-hidden-Markov-models
The project tackles developing a part-of speech (POS) tagger for Afrikaans, a South African language, using a hidden Markov model (HMM)by building a first-order HMM tagger that accurately assigns tags to words, employing a supervised approach to estimate the HMM parameters based on the training data.

# Datasets
The Afrikaans dataset used in this project was acquired from the [South African Centre for Digital Language Resources](https://repo.sadilar.org) (SADiLaR). Specifically, the dataset belongs to the NCHLT (National Centre for Human Language Technologies) Annotated Text Corpora, which is a collection of annotated corpora for various South African languages

# Training the basic first-order HMM tagger

### Model Training
Training the first-order HMM tagger involves estimating transition and emission probabilities, $P(\text{tag}_t|\text{tag}_{t-1})$ and  $P(\text{word}_t|\text{tag}_t)$, respectively based on the occurrences of the words and tags in the training data. See [our paper](https://drive.google.com/uc?id=1PO3UjJjw7FD3zUU4aRJssldiixl8DeRh) here for the full details on the training process.

### Dataset splitting
To evaluate the performance of our HMM tagger during training, we split the training set into 80/20 training and validation split. 

### POS tagging using the Viterbi algorithm
The Viterbi algorithm, a dynamic programming algorithm for finding the most probable sequence of hidden states, was implemented to decode part-of-speech tags for new sentences

# Results
The table below shows the accuracy of our HMM tagger on the validation and training sets.
|Evaluation Datasets|  Accuracy (%)|
|:------------------|:-------------|
|Validation set     |90.7528       |
|Test set           |90.6283       |

Table II below shows one of the sentences with a tag that was wrongly predicted. https://drive.google.com/file/d//view?usp=sharing
![alt text](https://drive.google.com/uc?id=12cK5Wf8KVoQD0ZMxeVrxZXoyJR-YHgwu)

# Running the code
### Dependencies
- Python 3.x
- Anaconda 3
- Jupyter Notebook
- Visual Studio Code ipynb
- Libraries: numpy, pandas, unicodedata, seaborn, collections.

### License
Refer to the [LICENSE](https://github.com/naftalindeapo/Language-Modeling-and-Byte-pair-Encoding-Project/blob/main/LICENSE).




