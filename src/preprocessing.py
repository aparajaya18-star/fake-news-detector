import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess(df, remove_stopwords=True, remove_sourcewords=False, remove_punctuation=True, lowercase=True, check= False):
    df = df.copy()
    # Converting datatypes
    df['title'] = df['title'].astype('string')
    df['text'] = df['text'].astype('string')

    # Treating missing values
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # Convert to Lowercase
    if lowercase:
        df['text'] = df['text'].str.lower()
        df['title'] = df['title'].str.lower()

    # Remove punctuation
    if remove_punctuation: 
        df['title'] = df['title'].str.replace(r'[^\w\s]+', '', regex=True) 
        df['text'] = df['text'].str.replace(r'[^\w\s]+', '', regex=True) 

    # Remove stopwords 
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))

        df['text'] = df['text'].apply(
            lambda x: " ".join([word for word in x.split() if word not in stop_words])
        )
        df['title'] = df['title'].apply(
            lambda x: " ".join([word for word in x.split() if word not in stop_words])
        )

    # Remove article source names
    if remove_sourcewords:
        source_phrases = [
            "washington reuters",
            "new york times",
            "associated press",
            "wall street journal",
            "fox news",
            "breitbart",
            "cnn",
            "bbc",
            "reuters"
        ]

        pattern = r'\b(?:' + '|'.join(map(re.escape, source_phrases)) + r')\b'

        df['text'] = df['text'].str.replace(pattern, '', regex=True)
        df['title'] = df['title'].str.replace(pattern, '', regex=True)

        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Check if pre-processing caused too many empty rows
    if check: 
        print("No of empty title rows: ",(df['title'].str.len() == 0).sum())
        print("No of empty text rows: ",(df['text'].str.len() == 0).sum())
        count = ((df['title'].str.len() == 0) & (df['text'].str.len() == 0)).sum()
        print("No of rows with empty title AND text: ",count)
    
    return df