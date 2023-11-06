import visualize_data
from scipy.spatial.distance import euclidean
import numpy as np

def custom_round(x):
    # np.where is used to apply a condition element-wise
    # For positive numbers halfway between two integers, round up
    # For negative numbers halfway between two integers, round down
    # Otherwise, use standard rounding
    return np.where((x - np.floor(x) == 0.5) & (x > 0), np.ceil(x),
            np.where((x - np.ceil(x) == -0.5) & (x < 0), np.floor(x),
                     np.round(x)))

def normal_line_through_midpoint(P1, P2):
    
    #print(f'!!! Calculating the m_perpendicular of the boundary between {P1} and {P2}... !!!')

    # Calculate the middle point
    M = [(P1[0] + P2[0]) / 2, (P1[1] + P2[1]) / 2]

    # Calculate the slope of the line connecting P1 and P2
    if (P2[0] - P1[0]) == 0:  # To avoid division by zero
        m_perpendicular = 0
    else:
        m = (P2[1] - P1[1]) / (P2[0] - P1[0])
        # Calculate the slope of the line perpendicular to the above line
        if m == 0:  # To avoid division by zero
            m_perpendicular = float('inf')  # Infinite slope represents a vertical line
        else:
            m_perpendicular = -1 / m

    # Compute the y-intercept of the normal line
    if m_perpendicular == float('inf'):
        b = 0
    else:
        b = M[1] - m_perpendicular * M[0]
    
    return M, m_perpendicular, b


def find_boundaries(X, CH, title):
    # Define the boundaries array with size cluster.size - 1  
    boundaries_points  = np.zeros(( CH.shape[0]-1, CH.shape[1]))
    boundaries_lines_m = [None] * (boundaries_points.shape[0])
    boundaries_lines_b = [None] * (boundaries_points.shape[0])

    if CH.shape[0] == 1:
        print('There are not boundaries to be found')
    else:
        for i in range(CH.shape[0]):
            
            # Find the boundary between the actual two points 
            #boundaries_points[i] = (CH[i+1] + CH[i]) / 2
            
            boundaries_points[i], boundaries_lines_m[i], boundaries_lines_b[i] = normal_line_through_midpoint(CH[i], CH[i+1])

            # print(f'boundary #{i+1} between cluster:{CH[i]} and cluster:{CH[i+1]} is: \n'
            #       f'point: {boundaries_points[i]}, m: {boundaries_lines_m[i]}, b: {boundaries_lines_b[i]} \n') 

            if i == CH.shape[0]-2:
                break
            
    visualize_data.plot_data(X, CH=CH, show_CHs=True, 
                            boundaries_points=boundaries_points, 
                            boundaries_lines_m=boundaries_lines_m, 
                            boundaries_lines_b=boundaries_lines_b, 
                            show_line=True, title=title)

    return boundaries_points, boundaries_lines_m, boundaries_lines_b


def allocate_datapoints(X, CH, boundaries_lines_m, boundaries_lines_b):
    
    # define a dictionary with each cluster with its corresponding points.
    cluster_points = {}

    # Assign the points to their corresponding clusters evaluating the boundary
    # !!! I will take the values that lay on the edge for the upper cluster !!!
    for i in range(CH.shape[0]):
        if i == 0:
            cluster_points[i] = [point 
                                for point in X 
                                if boundaries_lines_m[i]*point[0] + boundaries_lines_b[i] - point[1] > 0]
        elif i == CH.shape[0]-1:
            cluster_points[i] = [point 
                                for point in X 
                                if boundaries_lines_m[i-1]*point[0] + boundaries_lines_b[i-1] - point[1] <= 0]
        else:
            cluster_points[i] = [point 
                                for point in X 
                                if boundaries_lines_m[i-1]*point[0] + boundaries_lines_b[i-1] - point[1] <= 0
                                and boundaries_lines_m[i]*point[0] + boundaries_lines_b[i] - point[1] > 0]
    
    return cluster_points

def compute_D(CH, cluster_points):
    
    # Distance (Square residual D)
    D = 0

    for i in cluster_points:
        centroid = CH[i]
        for point in cluster_points[i]:
            distance = euclidean(point, centroid)**2
            D += distance
    
    return np.round(D)

def compute_centroids(X, CH, cluster_points, boundaries_points, boundaries_lines_m, boundaries_lines_b, title):

    # Find the new cluster positions using the average of their corresponding points
    for i in cluster_points.keys():
        CH[i] = custom_round(sum(cluster_points[i]) / len(cluster_points[i]))

    # Sort cluster array
    CH = CH.astype(int) # For eliminate the ".0" in the graph
    CH = np.sort(CH, axis=0)
    
    visualize_data.plot_data(X, CH=CH, show_CHs=True, 
                            boundaries_points=boundaries_points, 
                            boundaries_lines_m=boundaries_lines_m, 
                            boundaries_lines_b=boundaries_lines_b, 
                            show_line=True, title=title)

    return CH


def find_farest_node(CH, cluster_points):
    
    # The farest node of the entire dataset
    dist_max = 0
    dist_max_cluster = None

    for i in cluster_points:
        centroid = CH[i]
        for point in cluster_points[i]:
            distance = euclidean(point, centroid)
            if distance > dist_max:
                dist_max = distance # maximum distance from a point to its cluster
                dist_max_cluster = centroid # index of the cluster that has the farest point
                dist_max_point = point

                # Find the average and median distances from the points to the cluster
                distances = np.array([euclidean(point, centroid) for point in cluster_points[i]])
                avg_dist = np.mean(distances)

    print(f'The farest node {dist_max_point} has a distance of {dist_max} to the cluster {dist_max_cluster}')
    print(f'The average distance from the nodes to this cluster is {avg_dist}')
    

    return dist_max_cluster, dist_max_point, dist_max, avg_dist
        

def add_centroid(dist_max_cluster, dist_max_point, dist_max):

    # Find the direction where the new cluster has to be located following the direction of this farest node
    if (dist_max_cluster[0] - dist_max_point[0] > 0) and (dist_max_cluster[1] - dist_max_point[1] > 0) :    
        # move to the left
        new_CH = np.round(dist_max_cluster - ( (dist_max) / 2))
    else:
        # move to the right
        new_CH = np.round(dist_max_cluster + ( (dist_max) / 2))
    
    print(f'newCH = {new_CH}, last_CH = {dist_max_cluster}')

    return new_CH