# Basic Imports
import sys
import pandas as pd
import time
from pathlib import Path

project_root = Path.cwd().parent
sys.path.append(str(project_root))

# Importing shared functions
from src.preprocessing import preprocess
from src.vectorization import vectorize
from src.evaluation import evaluate

def run_experiment(df, model, vectorizer, experiment_name="", ngramRange=(1,2), remove_stopwords=True, vec_title=True, vec_text=True, mdf=5, remove_sourcewords=False):
    # Creating features
    start_time = time.perf_counter()

    df = preprocess(df,remove_stopwords=remove_stopwords, remove_sourcewords=remove_sourcewords)
    X_train, X_test, Y_train, Y_test, feature_names, total_features = vectorize(df, method = vectorizer, ngramRange=ngramRange, vec_title=vec_title, vec_text=vec_text, mdf=mdf)

    end_time = time.perf_counter()
    creation_time = end_time - start_time

    print(f"\nFeatures created in {creation_time:.4f} seconds.")
    print("Total features = ",total_features)

    # Training model
    start_time = time.perf_counter()

    model = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    end_time = time.perf_counter()
    training_time = end_time - start_time

    print(f"Training completed in {training_time:.4f} seconds.")
    
    # Evaluation
    accuracy, precision, recall, f1, cm, top_negative_features, top_positive_features = evaluate(Y_test, Y_pred, model, feature_names)

    return {
        "Experiment": experiment_name,
        "Vectorizer": vectorizer,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Features": total_features,
        "Feature Time (s)": creation_time,
        "Training Time (s)": training_time
    }