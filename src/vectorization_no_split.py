import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack

def vectorize_no_split(X, method, mdf=5, ngramRange=(1,2), max_features_title=50000, max_features_text=500000, vec_text = True, vec_title = True, verbose = False):

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
        matrix_title = vectorizer_title.fit_transform(X["title"])
        matrix_text = vectorizer_text.fit_transform(X["text"])

        X_train = hstack([matrix_title, matrix_text])

        feature_names = np.concatenate((
        vectorizer_title.get_feature_names_out(),
        vectorizer_text.get_feature_names_out()
        ))
    elif vec_title:
        # Fit and transform the data
        X_train = vectorizer_title.fit_transform(X["title"])
        feature_names = vectorizer_title.get_feature_names_out()
    elif vec_text:
        # Fit and transform the data
        X_train = vectorizer_text.fit_transform(X["text"])
        feature_names = vectorizer_text.get_feature_names_out()
    else:
        raise ValueError(
            "At least one of vec_title or vec_text must be True"
        )
    
    total_features = X_train.shape[1]
    return X_train, vectorizer_title, vectorizer_text, feature_names, total_features