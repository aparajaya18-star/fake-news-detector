# Vectorization
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack

def vectorize(df, method, mdf=5, ngramRange=(1,2), max_features_title=50000, max_features_text=500000, vec_text = True, vec_title = True, verbose = False):
    X = df[["title", "text"]]
    Y = df["label"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42)

    # Initialize the vectorizer
    if method == "tfidf":
        vectorizer_title = TfidfVectorizer(min_df=mdf, ngram_range=ngramRange, max_features=max_features_title)
        vectorizer_text = TfidfVectorizer(min_df=mdf, ngram_range=ngramRange, max_features=max_features_text)
    elif method == "count":
        vectorizer_title = CountVectorizer(min_df=mdf, ngram_range=ngramRange, max_features=max_features_title)
        vectorizer_text = CountVectorizer(min_df=mdf, ngram_range=ngramRange, max_features=max_features_text)
    else:
        raise ValueError(
            "method must be 'tfidf' or 'count'"
        )

    # Vectorize text and/or title
    if vec_text and vec_title:
        # Fit and transform the data
        matrix_title_train = vectorizer_title.fit_transform(X_train["title"])
        matrix_title_test = vectorizer_title.transform(X_test["title"])

        matrix_text_train = vectorizer_text.fit_transform(X_train["text"])
        matrix_text_test = vectorizer_text.transform(X_test["text"])

        X_train = hstack([matrix_title_train, matrix_text_train])
        X_test = hstack([matrix_title_test, matrix_text_test])

        feature_names = np.concatenate((
        vectorizer_title.get_feature_names_out(),
        vectorizer_text.get_feature_names_out()
        ))
    elif vec_title:
        # Fit and transform the data
        matrix_title_train = vectorizer_title.fit_transform(X_train["title"])
        matrix_title_test = vectorizer_title.transform(X_test["title"])
        
        X_train = matrix_title_train
        X_test = matrix_title_test

        feature_names = vectorizer_title.get_feature_names_out()
    elif vec_text:
        # Fit and transform the data
        matrix_text_train = vectorizer_text.fit_transform(X_train["text"])
        matrix_text_test = vectorizer_text.transform(X_test["text"])

        X_train = matrix_text_train
        X_test = matrix_text_test

        feature_names = vectorizer_text.get_feature_names_out()


    # Matrix inspection
    # Shape = (number_of_documents, number_of_features)
    if verbose:
        print(X_train.shape)
        print(X_test.shape)

    total_features = X_train.shape[1]
    return X_train, X_test, Y_train, Y_test, feature_names, total_features