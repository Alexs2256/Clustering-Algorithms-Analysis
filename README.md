# Clustering-Algorithms-Analysis
This project compares K-Means and Hierarchical Agglomerative Clustering algorithms using the silhouette coefficient for evaluation. It includes outlier detection and removal, and allows users to customize the clustering process on a synthetic 3D dataset to identify the superior algorithm based on clustering quality.

Overview
This project compares two popular clustering algorithms—K-Means and Hierarchical Agglomerative Clustering—by evaluating their performance using the silhouette coefficient. The silhouette coefficient is a measure of how well each point fits within its assigned cluster. The goal is to identify which algorithm performs better based on clustering quality, using a synthetic 3D dataset.

Features:
Outlier Detection and Removal: The project includes a method to detect and remove outliers from the dataset using a user-defined threshold.
K-Means Algorithm: A standard implementation of the K-Means clustering algorithm is used to partition the data into K clusters.
Hierarchical Agglomerative Clustering: This algorithm iteratively merges clusters based on various distance metrics, such as minimum, maximum, average, and center distance, to create hierarchical clusters.
Silhouette Coefficient Calculation: For each clustering method, the silhouette coefficient is calculated to evaluate the clustering quality. A higher silhouette score indicates better-defined clusters.
Customizable K Value: Users can specify the number of clusters (K) they wish to use for clustering.
Dataset:
A synthetic dataset of 500 points in a 3D space is generated. The points are randomly distributed and the dataset can be modified by the user to test different scenarios. The algorithm can handle various points in the dataset and will also remove outliers based on the user-specified parameters.

How It Works:
Outlier Detection: The program removes outliers based on the distance between points and a user-defined threshold.
K-Means Clustering: The K-Means algorithm is run on the dataset, and the silhouette coefficient is computed to measure the clustering quality.
Hierarchical Agglomerative Clustering: Four types of hierarchical clustering are implemented: Min Distance, Max Distance, Average Distance, and Center Distance. The silhouette coefficient is calculated for each method.
Comparison: The algorithm with the highest silhouette score is considered the superior clustering algorithm for the given dataset.
Usage:
Run the script and input the number of clusters (K) you wish to test.
The program will generate the dataset, perform outlier detection, and apply both clustering algorithms.
Afterward, the silhouette coefficients for each method will be displayed, and the algorithm with the best performance will be identified.
Requirements:
Python 3.x
numpy
random
math
