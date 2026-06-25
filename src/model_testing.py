import time
import numpy as np
from scipy.sparse import hstack
from src.preprocessing import preprocess
from src.evaluation import evaluate

def test_model(df_raw, vectorizer_title, vectorizer_text, model, text_only=False, title_only=False):
    # ----- Feature Creation -----#

    start_time = time.perf_counter()
    # Preprocessing
    df = preprocess(df_raw, check=True)

    # Vectorization
    X_test = df[["title", "text"]]
    Y_test = df["label"]

    if title_only and text_only:
        raise ValueError("Both text_only and title_only cannot be True")
    elif title_only:
        X_test = vectorizer_title.transform(X_test["title"])
        feature_names = vectorizer_title.get_feature_names_out()
    elif text_only:
        X_test = vectorizer_text.transform(X_test["text"])
        feature_names = vectorizer_text.get_feature_names_out()
    else:
        matrix_title_test = vectorizer_title.transform(X_test["title"])
        matrix_text_test = vectorizer_text.transform(X_test["text"])
        X_test = hstack([matrix_title_test, matrix_text_test])
        feature_names = np.concatenate((
            vectorizer_title.get_feature_names_out(),
            vectorizer_text.get_feature_names_out()
            ))
        
    end_time = time.perf_counter()
    creation_time = end_time - start_time

    total_features = X_test.shape[1]

    print(f"\nFeatures created in {creation_time:.4f} seconds.")
    print("Total features = ",total_features)
 
    # ------ Prediction ------ #
    Y_pred = model.predict(X_test)

    # ------ Evaluation ------ #
    accuracy, precision, recall, f1, cm, top_negative_features, top_positive_features = evaluate(Y_test, Y_pred, model, feature_names)

    return accuracy, precision, recall, f1, cm, top_negative_features, top_positive_features