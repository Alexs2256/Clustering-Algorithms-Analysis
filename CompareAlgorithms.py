import numpy
import math
import random
from KMeansAlgorithm import KMeansAlgorithm
from MeasureClustering import MeasureClustering
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class CompareAlgorithms:
    def __init__(self):
        self.cluster_type=""
    def removeOutliers(self):
        def DB(p, D):
            num_points = 500
            # Generate 500 points in 3D space
            # Each point will have 3 coordinates (x, y, z) generated uniformly between 0 and 1
            T = numpy.random.rand(num_points, 3) * 100 + 1
            outliers = []
            for i in range(len(T)):
                count = 0
                x1, y1, z1=T[i][0],T[i][1],T[i][2]
                for j in range(len(T)):
                    x2, y2, z2 = T[j][0], T[j][1], T[j][2]
                    distance=0
                    if j != i:
                        distance = math.sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))
                    if distance > D: count+=1
                if count >= p*num_points:
                    print("Outlier:", T[i])
                    outliers.append(tuple(T[i]))
            if len(outliers) == 0:
                print("No Outliers")
                return T.tolist()
            S = []
            for i in range(len(T)):
                if tuple(T[i]) not in outliers:
                    S.append(T[i])
            return S
        p,D=0,0
        while True:
            try:
                print("Please enter values for p (float) and D (integer):")
                p = float(input("Enter value for p: "))  # Input for float
                D = int(input("Enter value for D: "))  # Input for integer
                break
            except ValueError:
                print("Invalid input. Please make sure p is a float and D is an integer.")
        S = DB(p, D)
        return S
#-----------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    #K-means Algorithm Function
    def k_means_algorithm(self, K):
        def get_avg_silhoutte(K, function):
            avg_silhouette_Coefficient = 0
            d1 = function
            for c in range(len(d1)):
                if d1[c] == []: continue
                else:
                    s = MeasureClustering(d1, d1[c], c).silhouette_coefficient()
                    avg_silhouette_Coefficient+=s
            cluster_count=0
            for cluster in d1:
                if cluster!=[]: cluster_count+=1
            min_dist_sc = avg_silhouette_Coefficient/cluster_count
            return min_dist_sc
        k_means_obj = KMeansAlgorithm(S, K).k_means_algorithm()
        s = get_avg_silhoutte(K, k_means_obj)
        return s
#-----------------------------------------------------------------------------------------------------------------------
    #Heirarchical_agglomerative function
    def hierarchical_agglomerative(self, S, K):
        S1, S2, S3, S4 = S.copy(), S.copy(), S.copy(), S.copy()
        sample_data = [S1, S2, S3, S4]
        #Support Functions----------------------------------------------------------------------------------------------
        def distance(S, index1, index2):
            x1, y1, z1 = S[index1][0], S[index1][1], S[index1][2]
            x2, y2, z2 = S[index2][0], S[index2][1], S[index2][2]
            return math.sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))

        def distance_in_cluster(p1, p2):
            x1, y1, z1 = p1[0], p1[1], p1[2]
            x2, y2, z2 = p2[0], p2[1], p2[2]
            return math.sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))

        def list_to_array(linked_list):
            curr=linked_list
            arr=[]
            while curr != None:
                arr.append(curr.data)
                curr = curr.next
            return arr

        # Checks to see if a cluster is a Node or array
        def check_list(array, cluster_list):
            if type(cluster_list) == Node:
                array=list_to_array(cluster_list)
                return array
            else: return [cluster_list]

        def insertList(l1, l2):
            head = l1 if isinstance(l1, Node) else Node(l1)
            tail = l2 if isinstance(l2, Node) else Node(l2)
            curr = head
            while curr.next != None:
                curr=curr.next
            curr.next=tail
            return head

        def printList(ll):
            if ll is None:
                print("List is empty")
                return
            curr = ll
            while curr is not None:
                print(curr.data, end=" ")
                curr = curr.next
            print()  # For a newline after the list is printed

        def get_Cluster_Center(c):
            if type(c)==Node:
                cluster = list_to_array(c)
            else: cluster = [c]
            points = [0, 1, 2]
            cluster_center = [0, 0, 0]
            center_point_index = 0
            for point in points:
                sum = 0
                for coordinate in range(len(cluster)):
                    sum+=cluster[coordinate][point]
                cluster_center[center_point_index]=sum/len(cluster)
                center_point_index+=1
            return cluster_center

        #---------------------------------------------------------------------------------------------------------------
        #hierarchical agglomerative clusteringalgorithm
        #Min Clustering (Cluser based on Minimum distance between Clusters)
        def mergeLrgMinCluster(S, c1_index, c2_index):
            c1,c2, s_c1,s_c2 = S[c1_index], S[c2_index], [],[]
            smaller_Cluster,larger_Cluster = [],[]
            s_c1=check_list(s_c1, c1)
            s_c2=check_list(s_c2, c2)
            max_len = max(len(s_c1), len(s_c2))
            if len(s_c1) == max_len:
                larger_Cluster = s_c1
                smaller_Cluster = s_c2
            else:
                larger_Cluster = s_c2
                smaller_Cluster = s_c1
            min_dist_of_p = 100**3
            min_Cluster_Dist = 100**4
            for p1 in smaller_Cluster:
                for p2 in larger_Cluster:
                    d = distance_in_cluster(p1, p2)
                    min_dist_of_p = min(min_dist_of_p, d)
                min_Cluster_Dist=min(min_Cluster_Dist, min_dist_of_p)
            return min_Cluster_Dist

        def mergeMinDistanceClusters(K, S):
            cluster_type="Min Distance"
            clusters = len(S)
            while True:
                for c1 in range(len(S)):
                    if S[c1] is None or type(S[c1]) is None: continue
                    minimum = float('inf')
                    c2_index = -1
                    for c2 in range(len(S)):
                        if S[c2] is None or type(S[c2]) is None: continue
                        if c1 != c2:
                            if type(S[c1]) != Node and type(S[c2]) != Node:
                                d = distance(S, c1,c2)
                                if d < minimum:
                                    minimum = d
                                    c2_index = c2
                            else:
                                d = mergeLrgMinCluster(S, c1, c2)
                                if d < minimum:
                                    minimum = d
                                    c2_index = c2
                    merged_Cluster=insertList(S[c1], S[c2_index])
                    S[c1] = merged_Cluster
                    S[c2_index] = None
                    clusters-=1
                    if clusters == K:
                        break
                if clusters == K:
                    break

            return S
        #---------------------------------------------------------------------------------------------------------------
        #Max Clustering (Cluster Based on Longest Distance between Clusters)
        def mergeLrgMaxCluster(S, c1_index, c2_index):
            c1,c2, s_c1,s_c2 = S[c1_index], S[c2_index], [],[]
            smaller_Cluster,larger_Cluster = [],[]
            s_c1=check_list(s_c1, c1)
            s_c2=check_list(s_c2, c2)
            max_len = max(len(s_c1), len(s_c2))
            if len(s_c1) == max_len:
                larger_Cluster = s_c1
                smaller_Cluster = s_c2
            else:
                larger_Cluster = s_c2
                smaller_Cluster = s_c1
            max_dist_of_p = -1
            max_Cluster_Dist = -1
            for p1 in smaller_Cluster:
                for p2 in larger_Cluster:
                    d = distance_in_cluster(p1, p2)
                    max_dist_of_p = max(max_dist_of_p, d)
                max_Cluster_Dist=max(max_Cluster_Dist, max_dist_of_p)
            return max_Cluster_Dist

        def mergeMaxDistanceClusters(K, S):
            clusters = len(S)
            while True:
                for c1 in range(len(S)):
                    if S[c1] is None or type(S[c1]) is None: continue
                    maximum = -1
                    c2_index = -1
                    for c2 in range(len(S)):
                        if S[c2] is None or type(S[c2]) is None: continue
                        if c1 != c2:
                            if type(S[c1]) != Node and type(S[c2]) != Node:
                                d = distance(S, c1,c2)
                                if d > maximum:
                                    maximum = d
                                    c2_index = c2
                            else:
                                d = mergeLrgMaxCluster(S, c1, c2)
                                if d > maximum:
                                    maximum = d
                                    c2_index = c2
                    merged_Cluster=insertList(S[c1], S[c2_index])
                    S[c1] = merged_Cluster
                    S[c2_index] = None
                    clusters-=1
                    if clusters == K:
                        break
                if clusters == K:
                    break
            print("-----------------------------------------------------------------------------------------------------")
            return S
        #---------------------------------------------------------------------------------------------------------------
        #Average Clustering (Cluster based on minimum average distance between clusters)
        def mergeLrgAverageCluster(S, c1_index, c2_index):
            c1,c2, s_c1,s_c2 = S[c1_index], S[c2_index], [],[]
            smaller_Cluster,larger_Cluster = [],[]
            s_c1=check_list(s_c1, c1)
            s_c2=check_list(s_c2, c2)
            max_len = max(len(s_c1), len(s_c2))
            if len(s_c1) == max_len:
                larger_Cluster = s_c1
                smaller_Cluster = s_c2
            else:
                larger_Cluster = s_c2
                smaller_Cluster = s_c1
            min_dist_of_p = 100**3
            min_Cluster_Dist = 100**4
            sum_of_dist = 0
            for p1 in smaller_Cluster:
                for p2 in larger_Cluster:
                    d = distance_in_cluster(p1, p2)
                    sum_of_dist+=d
            return sum_of_dist/(len(p1)*len(p2))

        def mergeAverageDistanceClusters(K, S):
            clusters = len(S)
            while True:
                for c1 in range(len(S)):
                    if S[c1] is None or type(S[c1]) is None: continue
                    minimum = float('inf')
                    c2_index = -1
                    for c2 in range(len(S)):
                        if S[c2] is None or type(S[c2]) is None: continue
                        if c1 != c2:
                            if type(S[c1]) != Node and type(S[c2]) != Node:
                                d = distance(S, c1,c2)
                                if d < minimum:
                                    minimum = d
                                    c2_index = c2
                            else:
                                d = mergeLrgAverageCluster(S, c1, c2)
                                if d < minimum:
                                    minimum = d
                                    c2_index = c2
                    merged_Cluster=insertList(S[c1], S[c2_index])
                    S[c1] = merged_Cluster
                    S[c2_index] = None
                    clusters-=1
                    if clusters == K:
                        break
                if clusters == K:
                    break
            return S
        #---------------------------------------------------------------------------------------------------------------
        #Center Clustering (Cluster based on min distance between centers of clusters)
        def mergeLrgCenterCluster(S, c1_index, c2_index):
            c1, c2 = S[c1_index], S[c2_index]
            min_Cluster_Dist = float('inf')
            cen1, cen2 = get_Cluster_Center(c1), get_Cluster_Center(c2)
            return distance_in_cluster(cen1, cen2)

        def mergeCenterDistanceClusters(K, S):
            clusters = len(S)
            while True:
                for c1 in range(len(S)):
                    if S[c1] is None or type(S[c1]) is None: continue
                    minimum = float('inf')
                    c2_index = -1
                    for c2 in range(len(S)):
                        if S[c2] is None or type(S[c2]) is None: continue
                        if c1 != c2:
                            if type(S[c1]) != Node and type(S[c2]) != Node:
                                cen1,cen2 = get_Cluster_Center(S[c1]), get_Cluster_Center(S[c2])
                                d = distance_in_cluster(cen1,cen2)
                                if d < minimum:
                                    minimum = d
                                    c2_index = c2
                            else:
                                d = mergeLrgCenterCluster(S, c1, c2)
                                if d < minimum:
                                    minimum = d
                                    c2_index = c2
                    merged_Cluster = insertList(S[c1], S[c2_index])
                    S[c1] = merged_Cluster
                    S[c2_index] = None
                    clusters -= 1
                    if clusters == K:
                        break
                if clusters == K:
                    break

            return S
        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        #Calculate the average silhouette coefficient for a given set of clusters
        def get_avg_silhoutte(K, function, data):
            avg_silhouette_Coefficient=0
            d1 = function
            for c in range(len(d1)):
                if type(d1[c])!=numpy.ndarray and type(d1[c]) != type(None):
                    s=MeasureClustering(data, d1[c], c).silhouette_coefficient()
                    avg_silhouette_Coefficient+=s
            min_dist_sc=avg_silhouette_Coefficient/K
            return min_dist_sc
        #---------------------------------------------------------------------------------------------------------------
        #Get the best Heirarchical_agglomerative clustering method
        cluster_methods = [[mergeMinDistanceClusters(K, S1), "Min Distance Clustering"], [mergeMaxDistanceClusters(K, S2), "Max Distance Clustering"]
            ,[mergeAverageDistanceClusters(K, S3), "Average Distance Clustering"],[mergeCenterDistanceClusters(K, S4), "Center Distance Clustering"]]

        max_s=-1
        best_cluster_index=0
        for i in range(len(sample_data)):
            best_coefficient = get_avg_silhoutte(K,cluster_methods[i][0], sample_data[i])
            if best_coefficient > max_s:
                max_s=best_coefficient
                best_cluster_index=i
        best_clusters=cluster_methods[best_cluster_index][0]
        print("Best Heirarchical_agglomerative clustering method is: ", cluster_methods[best_cluster_index][1])
        for i in best_clusters:
            if type(i)==Node:printList(i)
            elif i != None: print(i)
        print("---------------------")
        print("Silhouette Coefficient:", best_coefficient)
        return best_coefficient, cluster_methods[best_cluster_index][1]
#End--------------------------------------------------------------------------------------------------------------------
while True:
    try:
        print("Please enter value for K: ")
        K = int(input())  # Try to convert input to an integer
        break  # Exit loop if the conversion is successful
    except ValueError:
        # Handle the case where input is not an integer
        print("Invalid input. Please enter an integer value for K.")
obj = CompareAlgorithms()
S = obj.removeOutliers()
#-----------------------------------------------------------------------------------------------------------------------
h_a_silhouette_coefficient=obj.hierarchical_agglomerative(S, K)
#-----------------------------------------------------------------------------------------------------------------------
print()
k_means_silhouette_coefficient=obj.k_means_algorithm(K)
print("Silhouette Coefficient:", k_means_silhouette_coefficient)
print()
print("---------------------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------
max_s=max(k_means_silhouette_coefficient, h_a_silhouette_coefficient[0])
if max_s == k_means_silhouette_coefficient:
    print("The superior algorithm is:", "K-Means Algorithm with a silhouette score of:", max_s)
else: print("The superior algorithm is:", "Hierarchical Agglomorative", h_a_silhouette_coefficient[1],
"with a silhouette score of:", max_s)
print("----------------------------------------------------")

