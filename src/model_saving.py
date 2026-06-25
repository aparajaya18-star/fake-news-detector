# Basic Imports
import joblib
import time

# Importing shared functions
from src.preprocessing import preprocess
from src.vectorization_no_split import vectorize_no_split

def run_and_save_model(df, model, vectorizer, model_name, ngramRange=(1,2), remove_stopwords=True, vec_title=True, vec_text=True, mdf=5, remove_sourcewords=False):
    # Creating features
    start_time = time.perf_counter()

    df = preprocess(df,remove_stopwords=remove_stopwords, remove_sourcewords=remove_sourcewords)
    
    # Vectorization
    X = df[["title", "text"]]
    Y = df["label"]

    X_train, vectorizer_title, vectorizer_text, feature_names, total_features = vectorize_no_split(X, method=vectorizer, mdf=mdf, ngramRange=ngramRange, vec_text = vec_text, vec_title = vec_title)

    end_time = time.perf_counter()
    creation_time = end_time - start_time

    print(f"\nFeatures created in {creation_time:.4f} seconds.")
    print("Total features = ",total_features)

    # Training model
    start_time = time.perf_counter()

    model = model.fit(X_train, Y)

    end_time = time.perf_counter()
    training_time = end_time - start_time

    print(f"Training completed in {training_time:.4f} seconds.")
    metadata = {
        "vectorizer": vectorizer,
        "ngram_range": ngramRange,
        "min_df": mdf,
        "remove_stopwords": remove_stopwords,
        "remove_sourcewords": remove_sourcewords,
        "vec_title": vec_title,
        "vec_text": vec_text,
        "total_features": total_features
    }

    joblib.dump(model, f"../models/{model_name}.pkl")
    joblib.dump(vectorizer_title, f"../models/{model_name}_title_vectorizer.pkl")
    joblib.dump(vectorizer_text, f"../models/{model_name}_text_vectorizer.pkl")
    joblib.dump(metadata, f"../models/{model_name}_metadata.pkl")
    
    return model, vectorizer_title, vectorizer_text, feature_names