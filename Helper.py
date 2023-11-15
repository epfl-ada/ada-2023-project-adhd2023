from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(text):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the input text and transform the text into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform([text])

    # Convert the TF-IDF matrix to a dense array and return the vectorized representation
    vectorized_text = tfidf_matrix.toarray()

    return vectorized_text

def load_csv(file_path):
    """Load data from a CSV file."""
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data