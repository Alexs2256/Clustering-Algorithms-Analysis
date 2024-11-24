
import math
import numpy
class MeasureClustering:
    def __init__(self, S, data, cluster_index):
        self.data=data
        self.cluster_index = cluster_index
        self.S=S
    def silhouette_coefficient(self):
        #---------------------------------------------------------------------------------------------------------------
        #Support Functions
        def distance_in_cluster(p1, p2):
            print(p1, p2)
            x1, y1, z1 = p1[0], p1[1], p1[2]
            x2, y2, z2 = p2[0], p2[1], p2[2]
            return math.sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))

        def list_to_array(linked_list):
            if type(linked_list)==numpy.ndarray or type(linked_list)==list: return [linked_list]
            curr=linked_list
            arr=[]
            while curr != None:
                arr.append(curr.data)
                curr = curr.next
            return arr
        #---------------------------------------------------------------------------------------------------------------
        def get_a1(cluster, point_index):
            if len(cluster)==1: return 0
            p1, sum = cluster[point_index], 0
            for point in range(len(cluster)):
                if point != point_index:
                    sum+=distance_in_cluster(p1, cluster[point])
            avg=sum/(len(cluster)-1)
            return avg

        def get_b1(S, cluster_index, point_index):
            if type(S)==numpy.ndarray: return 0
            p=list_to_array(S[cluster_index])[point_index]
            min_cluster = float('inf')
            for index, cluster in enumerate(S):
                if type(cluster) != type(None) and index != cluster_index:
                    cluster = list_to_array(cluster)
                    sum=0
                    for point in cluster:
                        sum += distance_in_cluster(p, point)
                    avg=sum/len(cluster)
                    min_cluster=min(min_cluster, avg)
            return min_cluster

        cluster = list_to_array(self.data)
        sum_S=0
        for point_index in range(len(cluster)):
            a1 = get_a1(cluster, point_index)
            b1 = get_b1(self.S, self.cluster_index, point_index)
            sum_S+=(b1-a1)/max(b1, a1)
        s=sum_S/len(cluster)
        return s












