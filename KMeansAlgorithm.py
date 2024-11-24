import numpy as np
import math
import random

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class KMeansAlgorithm:
    def __init__(self, S, K):
        self.S = S
        self.K=K
    def k_means_algorithm(self):
        # Support Functions
        #---------------------------------------------------------------------------------------------------------------
        def distance_in_cluster(p1, p2):
            x1, y1, z1 = p1[0], p1[1], p1[2]
            x2, y2, z2 = p2[0], p2[1], p2[2]
            return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2))

        def list_to_array(linked_list):
            curr=linked_list
            arr=[]
            while curr != None:
                arr.append(curr.data)
                curr = curr.next
            return arr

        def check_list(array, cluster_list):
            if type(cluster_list) == Node:
                array=list_to_array(cluster_list)
            else: return [cluster_list]
            return array

        def get_Cluster_Center(c):
            cluster=[]
            cluster=check_list(cluster, c)
            points = [0, 1, 2] #[x, y, z]
            cluster_center = [0, 0, 0]
            center_point_index = 0
            for point in points:
                sum = 0
                for coordinate in range(len(cluster)):
                    sum+=cluster[coordinate][point]
                cluster_center[center_point_index]=sum/len(cluster)
                center_point_index+=1
            return cluster_center

        def insertList(l1, l2):
            if l1 == []:
                head = Node(l2)
                return head
            head = l1 if isinstance(l1, Node) else Node(l1)
            tail = l2 if isinstance(l2, Node) else Node(l2)

            curr = head
            while curr.next != None:
                curr = curr.next
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
        #---------------------------------------------------------------------------------------------------------------
        def partition(K):
            partitions = []
            taken_indices = set()
            for i in range(0, K):
                random_center_index = random.randint(0, len(self.S) - 1)
                if random_center_index not in taken_indices:
                    taken_indices.add(random_center_index)
                    partitions.append(self.S[random_center_index])
            return partitions

        def matching_centroids(center1, center2):
            return center1==center2

        def cluster(partitions, S):
            prev_centroids = [None]*len(partitions)
            iteration = 0
            while True:
                curr_partition = [[]]*self.K
                for point in S:
                    closest_center=100**3
                    center_index = -1
                    for center in range(len(partitions)):
                        d = distance_in_cluster(point, partitions[center])
                        if d < closest_center:
                            closest_center=d
                            center_index=center
                    merged_Cluster = insertList(curr_partition[center_index], point)
                    curr_partition[center_index] = merged_Cluster
                for i in range(len(partitions)):
                    partitions[i] = get_Cluster_Center(curr_partition[i])
                if matching_centroids(prev_centroids, partitions):
                    print("K-Means algorithm clusters:")
                    for i in curr_partition:
                        if type(i)==Node:printList(i)
                        else: print(i)
                    print("------------------------------")
                    return curr_partition
                #If previous centers don't match with current centers
                #copy current centers to previous and continue
                for i in range(len(partitions)):
                    prev_centroids[i]=get_Cluster_Center(curr_partition[i])

        c = cluster(partition(self.K), self.S)
        for i in c:
            if i ==[]: c.remove(i)
        return c












