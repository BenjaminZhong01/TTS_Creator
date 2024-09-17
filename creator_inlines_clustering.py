import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Download NLTK data (uncomment if necessary)
# nltk.download('stopwords', download_dir='/Users/bytedance/PycharmProjects')

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
    # Filter complaints with label 'useful'
    return df[df['label'].str.lower() == 'useful']['complaint']

# Preprocess text data
def preprocess_text(text):
    # Convert to lowercase, remove stopwords
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

# Vectorize the complaints using TF-IDF
def vectorize_complaints(complaints):
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(complaints)
    return X, vectorizer

# Perform clustering using KMeans
def cluster_complaints(X, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

# Extract top terms from each cluster
def extract_topics(kmeans, vectorizer, n_terms=10):
    terms = vectorizer.get_feature_names_out()
    topics = []
    for idx, cluster_center in enumerate(kmeans.cluster_centers_):
        topic_terms = [terms[i] for i in cluster_center.argsort()[-n_terms:]]
        topics.append("Cluster {}: {}".format(idx, ', '.join(topic_terms)))
    return topics

# Visualize clusters using PCA
def visualize_clusters(X, kmeans):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X.toarray())
    labels = kmeans.labels_

    # Plotting
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='rainbow')
    plt.title('Complaint Clusters Visualization')
    plt.show()

# Main function to process the complaints
def main(file_path):
    complaints = load_complaints(file_path)
    complaints = complaints.apply(preprocess_text)

    X, vectorizer = vectorize_complaints(complaints)

    # Perform KMeans clustering
    num_clusters = 5  # You can tweak this value
    kmeans = cluster_complaints(X, num_clusters=num_clusters)

    # Add cluster labels to the DataFrame
    df = pd.read_csv(file_path)
    df = df[df['label'].str.lower() == 'useful']  # Filter 'useful' complaints again
    df['processed_complaint'] = df['complaint'].apply(preprocess_text)
    df['cluster'] = kmeans.predict(vectorizer.transform(df['processed_complaint']))

    # Output topics for each cluster
    topics = extract_topics(kmeans, vectorizer)
    for topic in topics:
        print(topic)

    # Optional: Visualize the clusters
    # visualize_clusters(X, kmeans)

    # Save the DataFrame with the cluster labels to a new CSV file
    output_file_path = file_path.replace(".csv", "_with_clusters.csv")
    df.to_csv(output_file_path, index=False)
    print(f"Updated CSV saved to {output_file_path}")

# Example usage
file_path = '/Users/bytedance/Downloads/classified_complaints - Sheet1.csv'  # Replace with your file path
main(file_path)

