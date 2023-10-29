import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

# Category Names
category_names = {0: "Fire", 1: "Crime", 2: "Health"}

# Sample Data
data = {
    "Description": [
        "There was a fire in the chemistry lab at the university.",
        "A theft occurred in the electronics store last night.",
        "A student had a medical emergency during a class.",
        "I witnessed a hit-and-run accident on Main Street.",
        "There was a large fight at the local bar involving multiple people.",
        "A car crashed into a tree in the park.",
        "Someone reported a suspicious package at the train station.",
        "A person was assaulted in the park in the evening.",
        "I found an injured bird in my backyard.",
        "There's a gas leak in the apartment building.",
        "A burglary took place at my neighbor's house.",
        "I saw a person who fainted on the subway platform.",
        "A dog is stuck in a tree in the park.",
        "A drunk driver was seen swerving on the highway.",
        "A building is on fire in the industrial area.",
        "I spotted a missing child at the shopping mall.",
        "A car was stolen from the parking lot of the grocery store.",
        "There's a power outage in the neighborhood.",
        "A fight broke out at a soccer game.",
        "I heard gunshots in the neighborhood last night.",
    ],
    "IncidentCategory": [0, 1, 2, 1, 1, 2, 1, 1, 2, 0, 1, 2, 2, 1, 0, 1, 1, 0, 0, 2],
}

df = pd.DataFrame(data)

# Text preprocessing and Feature Extraction
df["Description"] = (
    df["Description"].str.replace(r"[^\w\s]", "").str.lower()
)  # Convert to lowercase
df["Description"] = df["Description"].str.replace(r"[^\w\s]", "")  # Remove punctuation

# Tokenization
df["Description"] = df["Description"].apply(nltk.word_tokenize)

# Stop Word Removal
stop_words = set(stopwords.words("english"))
df["Description"] = df["Description"].apply(
    lambda tokens: [word for word in tokens if word not in stop_words]
)

# Stemming
stemmer = PorterStemmer()
df["Description"] = df["Description"].apply(
    lambda tokens: [stemmer.stem(word) for word in tokens]
)

# Join the tokens back into a single string
df["Description"] = df["Description"].apply(' '.join)

print(df["Description"])

# TF-IDF vectorization
print("Vectorising the text...")
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Description"])

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix,
    df["IncidentCategory"],
    test_size=0.2,
    random_state=42,
)

# Train the SVM Model
print("Training the model...")
svm_classifier = SVC(kernel="linear", C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)
print("Training complete!")

# Make predictions on the test set
predictions = svm_classifier.predict(X_test)

# Evaluate the model
report = classification_report(y_test, predictions)
print(report)

# Test user provided response.
user_description = "I saw a person who fainted on the subway platform."

# Preprocess the user's description to match the format used during training
user_description = user_description.lower()  # Convert to lowercase
user_description = user_description.replace(r"[^\w\s]", "")  # Remove punctuation

# Vectorize the user's description using the same TF-IDF vectorizer used during training
print("Vectorising the user's description...")
user_description_vector = tfidf_vectorizer.transform([user_description])

# Predict the incident category using the trained model
print("Predicting the incident category...")
predicted_category = svm_classifier.predict(user_description_vector)

# Map the category label to the actual category name
predicted_category_name = category_names[predicted_category[0]]

# Display the prediction
print(
    "Predicted Incident Category:",
    f"{predicted_category_name} ({predicted_category[0]})",
)
