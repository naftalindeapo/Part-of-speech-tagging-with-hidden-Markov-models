# Part-of-speech-tagging-with-hidden-Markov-models
The project tackles developing a part-of speech (POS) tagger for Afrikaans, a South African language, using a hidden Markov model (HMM)by building a first-order HMM tagger that accurately assigns tags to words, employing a supervised approach to estimate the HMM parameters based on the training data.

# Datasets
The Afrikaans dataset used in this project was acquired from the [South African Centre for Digital Language Resources](https://repo.sadilar.org) (SADiLaR). Specifically, the dataset belongs to the NCHLT (National Centre for Human Language Technologies) Annotated Text Corpora, which is a collection of annotated corpora for various South African languages

# Language modelling

### Training
In our language modeling approach, we estimated the probabilities of trigrams using the Maximum Likelihood Estimation (MLE) method via trigram model. See [our paper](https://drive.google.com/uc?id=1O3Wfa4Ulr-2_qnPZLo4DsXIto-gh3Guw) here for the full details on the training process.

### Text generation
We used the probabilities estimated from the training data to generate text from our trigram models to construct new sentences or text. Starting with an initial seed or context consisting of the first two characters, we iteratively select the next character based on the conditional probabilities of trigrams by choosing the character with the highest probability given the previous two characters until the desired length of the generated text is reached.

### Language model evaluation: 
We used the metric called *perplexity* to evaluate the performance of our trigram language model on the validation set for each of the five languages.

### Language identification
We used our trained trigram models to perform language identification on the test set by calculating the perplexity of each sentence under each language model. The language with the lowest perplexity is selected as the predicted language for the sentence. 

### Byte-pair encoding for language similarity
In addition to using language model perplexity to compare similarities between the five languages, we also applied a subword token learner to the data from each language and then considered the overlap between subword types. For this task, we used the bype-pair encode (BPE) algorithm.

# Results

### Trigrams learning
Table I below displays the trigrams and their probability with the two-character history ‘th’ for the English model.
![alt text](https://drive.google.com/uc?id=1N4mC47euB6cifQjAM_Fod5z1-gHh8Ltm) 

### Comparing Language similarity using perplexity
We used our trained models to calculate the cross-lingual perplexity measurements on the validation set and the results are shown below:
|         | Afrikaans| Dutch    |English   | Xhosa    | Zulu      |
|:--------|:---------|:---------|:---------|:---------|:----------|
|Afrikaans| 7.949203 | 11.444800| 18.307241| 43.649527| 45.656553 |
|Dutch    | 10.646834| 7.742023 | 15.234679| 52.431690| 54.124213 |
|English  | 20.820466| 22.214238| 7.659873 | 51.075642| 55.579143 |
|Xhosa    | 29.047159| 30.451402| 16.607398| 8.152562 | 8.909480  |
|Zulu     | 32.384335| 33.430517| 20.066052| 9.564616 | 8.489959  |

### Language identification of the training set
We used our trained models to perform language identification on the test set and the results are shown in Figure 1 below:
![alt text](https://drive.google.com/uc?id=1hFHidPJWcLqiEySl9oOq56-LDohvTfcW) 

### BPE learning
We applied the BPE algorithm to each training set for 100 merge iterations and computed the overlap of the BPE subword vocabulary between each of the languages. The results are shown in Table III and Figure 2 below.
<div>
    <img src="https://drive.google.com/uc?id=1kyQzzBliH1cLvyRYvQwZhVJbKybtfQCl" style="width: 48%; float: left;" /> 
    <img src="https://drive.google.com/uc?id=1qUtGTxs-ZwcTBLOnATuwBVeoXe-Kkarj" style="width: 40%; float: right;" />  
 </div>

# Running the code

### Dependencies
- Python 3.x
- Anaconda 3
- Jupyter Notebook
- Visual Studio Code ipynb
- Libraries: numpy, pandas, unicodedata, seaborn, collections, matplotlib, sklearn.

### License
Refer to the [LICENSE](https://github.com/naftalindeapo/Language-Modeling-and-Byte-pair-Encoding-Project/blob/main/LICENSE).




