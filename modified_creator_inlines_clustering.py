import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import matplotlib.pyplot as plt

# A simple stopwords list (without nltk)
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])


# Load complaints data from CSV
def load_complaints(file_path):
    df = pd.read_csv(file_path)
    return df[df['label'].str.lower() == 'useful']['complaint']


# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text


# Vectorize the complaints using TF-IDF
def vectorize_complaints(complaints):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(complaints)
    return X, vectorizer


# Function to find the best number of clusters using the Elbow method and Silhouette score
def find_best_num_clusters(X, max_clusters=10):
    wcss = []
    silhouette_scores = []

    # Try KMeans with 2 to max_clusters clusters
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)  # Sum of squared distances to closest cluster center
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))  # Silhouette score for each n_clusters

    # # Plot the WCSS (Elbow Method)
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(2, max_clusters + 1), wcss, marker='o', linestyle='--', color='b')
    # plt.title('Elbow Method - WCSS for each number of clusters')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()
    #
    # # Plot the Silhouette Scores
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='--', color='g')
    # plt.title('Silhouette Score for each number of clusters')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')
    # plt.show()

    # Return the number of clusters with the best silhouette score
    best_num_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 because index starts at 0
    print(f'Best number of clusters based on silhouette score: {best_num_clusters}')
    return best_num_clusters


# Perform clustering using KMeans
def cluster_complaints(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans


# Calculate the distance to the cluster center for each point
def calculate_distances(X, kmeans):
    closest, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)
    return distances


# Find the complaint closest to the cluster center
def get_closest_complaints(df, kmeans, vectorizer):
    X_transformed = vectorizer.transform(df['processed_complaint'])
    distances = calculate_distances(X_transformed, kmeans)
    df['distance_to_center'] = distances

    closest_complaints = []
    for cluster_num in range(kmeans.n_clusters):
        cluster_data = df[df['cluster'] == cluster_num]
        closest_idx = cluster_data['distance_to_center'].idxmin()
        closest_complaint = cluster_data.loc[closest_idx, 'complaint']
        closest_complaints.append((cluster_num, closest_complaint))

    return closest_complaints


# Main function to process the complaints
def main(file_path):
    complaints = load_complaints(file_path)
    complaints = complaints.apply(preprocess_text)

    X, vectorizer = vectorize_complaints(complaints)

    # Automatically determine the best number of clusters
    best_num_clusters = find_best_num_clusters(X, max_clusters=10)

    # Perform KMeans clustering using the best number of clusters
    kmeans = cluster_complaints(X, num_clusters=best_num_clusters)

    # Add cluster labels to the DataFrame
    df = pd.read_csv(file_path)
    df = df[df['label'].str.lower() == 'useful']
    df['processed_complaint'] = df['complaint'].apply(preprocess_text)
    df['cluster'] = kmeans.predict(vectorizer.transform(df['processed_complaint']))

    # Get the complaint closest to the center of each cluster
    closest_complaints = get_closest_complaints(df, kmeans, vectorizer)

    for cluster_num, complaint in closest_complaints:
        print(f"Cluster {cluster_num} closest complaint: {complaint}")

    # Save the DataFrame with the cluster labels and distances to a new CSV file
    output_file_path = file_path.replace(".csv", "_with_clusters_and_closest_complaints.csv")
    df.to_csv(output_file_path, index=False)
    print(f"Updated CSV saved to {output_file_path}")


# Example usage
file_path = 'classified_complaints.csv'
main(file_path)
