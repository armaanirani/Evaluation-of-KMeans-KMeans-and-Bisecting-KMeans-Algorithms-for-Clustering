import pandas as pd
import numpy as np
np.random.seed(21)
import matplotlib.pyplot as plt

def createSyntheticDataset(num_samples, num_features, num_clusters):
    """
    Generates a synthetic dataset with a given number of samples, features, and clusters.

    Args:
    - num_samples (int): The number of samples to generate.
    - num_features (int): The number of features per sample.
    - num_clusters (int): The number of distinct clusters.

    Returns:
    - pandas.DataFrame: A DataFrame containing the synthetic dataset with features.
    """
    # Generating random centers for each cluster within the range [-10, 10] for each feature.
    centers = [np.random.uniform(-10, 10, num_features) for _ in range(num_clusters)]

    # Initializing an empty list to store the samples from each cluster.
    cluster_data = []

    # Generating samples around each cluster center.
    for i in range(num_clusters):
        # Generating samples normally distributed around the center.
        samples = np.random.randn(num_samples // num_clusters, num_features) + centers[i]
        cluster_data.append(samples)

    # Concatenating all samples from different clusters into one array.
    all_samples = np.vstack(cluster_data)

    # Shuffling the data to mix the data points.
    np.random.shuffle(all_samples)

    # Creating a DataFrame from the generated data with column names 'feature_0', 'feature_1', etc.
    df = pd.DataFrame(all_samples, columns=[f'feature_{i}' for i in range(num_features)])

    return df

def computeDistance(point1, point2):
    """
    Computes the Euclidean distance between two points.

    Args:
    - point1 (numpy.ndarray): The first point represented as a NumPy array.
    - point2 (numpy.ndarray): The second point represented as a NumPy array.

    Returns:
    - float: The Euclidean distance between the two points.
    """
    # Calculatng the squared differences between the coordinates of the two points.
    squaredDiff = (point1 - point2) ** 2

    # Summing the squared differences to get the squared Euclidean distance.
    squaredDistance = np.sum(squaredDiff)

    # Taking the square root to get the Euclidean distance.
    distance = np.sqrt(squaredDistance)

    return distance

def initialSelection(data, k):
    """
    Randomly selects a subset of k samples from the input data.

    Args:
    - data (pandas.DataFrame): The input DataFrame containing the data.
    - k (int): The number of samples to be selected.

    Returns:
    - pandas.DataFrame: A new DataFrame containing k randomly selected samples.
    """
    # Randomly selecting k samples from the input data and reset the index of the resulting DataFrame.
    initSelec = data.sample(n=k).reset_index(drop=True)

    return initSelec

def assignClusterIds(data, centroids):
    """
    Assigns cluster IDs to each data point based on the closest centroid.

    Args:
    - data (pandas.DataFrame): The input DataFrame containing the data points.
    - centroids (pandas.DataFrame): The DataFrame containing the centroids.

    Returns:
    - numpy.ndarray: An array containing the cluster IDs for each data point.
    """
    # Computing the distance between each data point and each centroid.
    distances = np.array([[computeDistance(data.iloc[i], centroids.iloc[j]) 
                           for j in range(len(centroids))] 
                          for i in range(len(data))])

    # For each data point find the index of the closest centroid.
    clusterIds = np.argmin(distances, axis=1)

    return clusterIds

def computeClusterRepresentatives(data, clusterIds, k):
    """
    Computes the representative (mean) point for each cluster.

    Args:
    - data (pandas.DataFrame): The input DataFrame containing the data points.
    - clusterIds (numpy.ndarray): An array containing the cluster IDs for each data point.
    - k (int): The number of clusters.

    Returns:
    - pandas.DataFrame: A DataFrame containing the representative points for each cluster.
    """
    # Assigning the cluster IDs to the data DataFrame.
    grouped = data.assign(cluster_id=clusterIds).groupby('cluster_id')

    # Computing the mean of each feature for each cluster.
    clusterRepresentatives = grouped.mean().reset_index(drop=True)

    return clusterRepresentatives

def kMeans(data, k, maxIter=100):
    """
    Performs the K-Means clustering algorithm on the input data.

    Args:
    - data (pandas.DataFrame): The input DataFrame containing the data points.
    - k (int): The number of clusters to create.
    - maxIter (int): The maximum number of iterations to perform. Default is 100.

    Returns:
    - tuple: A tuple containing two elements:
        - pandas.DataFrame: A DataFrame containing the final centroids.
        - numpy.ndarray: An array containing the cluster IDs for each data point.

    Raises:
    - ValueError: If the input data is empty or if the number of clusters exceeds the number of data points.
    """
    # Checking if the input data is empty.
    if data.empty:
            raise ValueError("Input data to KMeans is empty.")
    # Checking if the number of clusters exceeds the number of data points.
    if k > len(data):
            raise ValueError(f"Number of clusters (k={k}) cannot exceed the number of data points ({len(data)}).")

    # Randomly initializing centroids.
    centroids = initialSelection(data, k)

    for _ in range(maxIter):
        # Assigning each data point to the closest centroid.
        clusterIds = assignClusterIds(data, centroids)

        # Calculating new centroids
        newCentroids = computeClusterRepresentatives(data, clusterIds, k)

        # Checking for convergence (if centroids do not change).
        if newCentroids.equals(centroids):
            break

        # Updating the centroids for the next iteration.
        centroids = newCentroids

    return centroids, clusterIds

def computeSilhouetteScores(data, kRange):
    """
    Computes the silhouette scores for the given range of k values.

    Args:
    - data (pandas.DataFrame): The input DataFrame containing the data points.
    - kRange (list or range): A list or range of k values to evaluate.

    Returns:
    - list: A list of silhouette scores, one for each k value in the provided range.
    """
    silhouetteScores = []

    for k in kRange:
        if k == 1:
            # Silhouette score is not defined for k=1.
            silhouetteScores.append(0)
            continue

        # Performing K-Means clustering and obtain cluster assignments.
        _, clusterIds = kMeans(data, k)

        # Computing the mean intra-cluster distances (a).
        a = np.array([np.mean([computeDistance(data.iloc[i], data.iloc[j]) 
                               for j in range(len(data)) 
                               if clusterIds[i] == clusterIds[j]])
                      for i in range(len(data))])

        # Computing the mean nearest-cluster distances (b).
        b = np.array([np.min([np.mean([computeDistance(data.iloc[i], data.iloc[j]) 
                                       for j in range(len(data)) 
                                       if clusterIds[j] == clusterId])
                              for clusterId in range(k) 
                              if clusterId != clusterIds[i]])
                      for i in range(len(data))])

        # Computing the silhouette values.
        silhouetteValues = (b - a) / np.maximum(a, b)

        # Computing the mean silhouette score for the current k value.
        silhouetteScores.append(np.mean(silhouetteValues))

    return silhouetteScores

def plot_silhouette(kRange, silhouetteScores):
    """
    Plots the silhouette scores for a range of k values.

    Args:
    - kRange (list or range): A list or range of k values for which the silhouette scores were computed.
    - silhouetteScores (list): A list of silhouette scores, one for each k value in kRange.
    """
    # Creating a new figure with a specific size.
    plt.figure(figsize=(10, 7))

    # Plotting the silhouette scores against the k values.
    plt.plot(kRange, silhouetteScores, marker='o')

    # Setting the title of the plot.
    plt.title("Silhouette Coefficient for Number of Clusters")

    # Setting the x-axis label.
    plt.xlabel("Number of Clusters (k)")

    # Setting the y-axis label.
    plt.ylabel("Silhouette Coefficient")

    # Adding a grid to the plot.
    plt.grid(True)

    # Saving the figure as a PNG file.
    plt.savefig('synthetic_silhouette_scores.png', format='png')

def main():
    """
    The main function that runs the complete K-Means clustering and silhouette score evaluation process.

    This function performs the following steps:
    1. Creates a synthetic dataset with the same number of data points as the 'dataset' (327).
    2. Computes the silhouette scores for a range of k values from 1 to 9.
    3. Plots the silhouette scores against the corresponding k values.
    """
    # Generating a synthetic dataset like the one provided.
    synthetic_data = createSyntheticDataset(327, 300, 2)

    # Defining the range of k values to evaluate.
    k_range = range(1, 10)

    # Computing the silhouette scores for the given range of k values.
    silhouette_scores = computeSilhouetteScores(synthetic_data, k_range)

    # Plotting the silhouette scores against the corresponding k values.
    plot_silhouette(k_range, silhouette_scores)

if __name__ == '__main__':
    main()