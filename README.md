# Fake News Classification on WELFake Dataset
### About the dataset:
* WELFake is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news.
* Dataset contains four columns: 
    - Serial number (starting from 0)
    - Title (about the text news heading)
    - Text (about the news content)
    - and Label (0 = fake and 1 = real).
* There are 78098 data entries in csv file out of which only 72134 entries are accessed as per the data frame.
* Published in: IEEE Transactions on Computational Social Systems: pp. 1-13 (doi: 10.1109/TCSS.2021.3068519).

## Possible Experiments
* Test TF-IDF vs Count-Vectorizer vs n-grams
* Experiment with above's parameters
* Try different word-embeddings
* Logistic Regression vs Multinomial Naive Bayes vs SVM
* With vs without stopwords
* Feature engineering (Process text or title only OR combine into singular feature etc)
* Transformer/BERT (optional stretch goal)

## Research:
* NLP
* Vectorization
    - One-Hot
    - Bag of Words
    - TF-IDF
    [ It is a popular technique in Natural Language Processing (NLP) to transform text into numerical features. It measures the importance of a word in a document relative to a collection of documents (corpus). ]
    - Word Embeddings
    - N-grams