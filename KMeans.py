import pandas as pd
import numpy as np
np.random.seed(21)
import matplotlib.pyplot as plt

def load_dataset(filename):
    """
    Reads a dataset from the given file name, formats the columns, and handles possible errors.

    Args:
    - filename (str): The path to the file containing the dataset.

    Returns:
    - pandas.DataFrame: The dataset loaded into a DataFrame with formatted columns.

    Raises:
    - ValueError: If no filename is given and/or dataset contains fewer than the minimum required data points or other input issues.
    - FileNotFoundError: If the file cannot be found.
    - IOError: If there's an error reading the file (including corruption).
    """
    # Checks if a filename is provided.
    if not filename:
        raise ValueError("No filename provided.")

    try:
        # Reads the file into a DataFrame.
        df = pd.read_csv(filename, sep=" ", header=None)

        # Checks if there's at least one data point.
        if df.shape[0] < 2:
            raise ValueError("Dataset must contain at least two data points.")

        # Setting the first column as 'label' as it contains the labels.
        df = df.rename(columns={0: 'label'})

        # Renaming the feature columns.
        numOfFeatures = df.shape[1] - 1  # Total columns minus one for the label.
        featureNames = {i: f'feature_{i-1}' for i in range(1, numOfFeatures + 1)}
        df = df.rename(columns=featureNames)
        
        # Removing the label column and retaining only features.
        dfFeatures = df.drop('label', axis=1)

    except FileNotFoundError:
        # Handles the case when the file cannot be found.
        raise FileNotFoundError(f"The file {filename} cannot be found.")
    except pd.errors.EmptyDataError:
        # Handles the case when the file is empty.
        raise ValueError("No data found in the file.")
    except Exception as e:
        # Handles any other exceptions that may occur while reading the file.
        raise IOError(f"An error occurred while reading the file: {str(e)}")

    return dfFeatures

def computeDistance(point1, point2):
    """
    Computes the Euclidean distance between two points.

    Args:
    - point1 (numpy.ndarray): The first point represented as a NumPy array.
    - point2 (numpy.ndarray): The second point represented as a NumPy array.

    Returns:
    - float: The Euclidean distance between the two points.
    """
    # Calculating the squared differences between the coordinates of the two points.
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
    # Checks if the input data is empty.
    if data.empty:
            raise ValueError("Input data to KMeans is empty.")
    # Checks if the number of clusters exceeds the number of data points.
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

        # Performing K-Means clustering and obtaining cluster assignments.
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
    plt.savefig('kmeans_silhouette_scores.png', format='png')

def main():
    """
    The main function that runs the complete K-Means clustering and silhouette score evaluation process.

    This function performs the following steps:
    1. Reads the dataset from a file named 'dataset'.
    2. Computes the silhouette scores for a range of k values from 1 to 9.
    3. Plots the silhouette scores against the corresponding k values.
    """
    # Reading the dataset from a file named 'dataset'.
    data = load_dataset('dataset')

    # Defining the range of k values to evaluate.
    k_range = range(1, 10)

    # Computing the silhouette scores for the given range of k values.
    silhouette_scores = computeSilhouetteScores(data, k_range)

    # Plotting the silhouette scores against the corresponding k values.
    plot_silhouette(k_range, silhouette_scores)

if __name__ == '__main__':
    main()