from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "This is a sample document.",
    "Another document for demonstration purposes.",
    "A third document to show how TF-IDF works.",
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (terms)
feature_names = vectorizer.get_feature_names_out()

# Display the TF-IDF matrix
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Display the feature names
print("\nFeature Names:")
print(feature_names)
