import visualize_data
from scipy.spatial.distance import euclidean
import numpy as np

# Function to generate a random set of n points with shape (n, 2) within a specified range
def generate_random_points(n, lower_bound, upper_bound):
    # Generate random points in the specified range [lower_bound, upper_bound)
    points = np.random.randint(low=lower_bound, high=upper_bound, size=(n, 2))
    return points

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
    
    # !!!!!!!!!! Modificate the criterium of the points that lie on the bloundary line !!!!!!!!

    # define a dictionary with each cluster with its corresponding points.
    cluster_points = {}

    # Assign the points to their corresponding clusters evaluating the boundary
    # !!! I will take the values that lay on the edge for the upper cluster !!!
    random_number = np.random.randint(2) # this is used for points that lie on the decision boundary, depend on the 
                                         # random number (0 or 1) in each iteration point will be in the (upper --> 1)
                                         # or (bottom --> 0) cluster.
    if random_number == 0: # (Bottom cluster)
        for i in range(CH.shape[0]):
            if i == 0: # Points in the first cluster (this cluster doesn't have bottom boundary)
                cluster_points[i] = [point 
                                    for point in X 
                                    if boundaries_lines_m[i]*point[0] + boundaries_lines_b[i] - point[1] > 0]
            elif i == CH.shape[0]-1: # Point in the last cluster (this cluster doesn't have top boundary)
                cluster_points[i] = [point 
                                    for point in X 
                                    if boundaries_lines_m[i-1]*point[0] + boundaries_lines_b[i-1] - point[1] <= 0]
            else:        
                    cluster_points[i] = [point # Points in the middle 
                                        for point in X 
                                        if boundaries_lines_m[i-1]*point[0] + boundaries_lines_b[i-1] - point[1] < 0
                                        and boundaries_lines_m[i]*point[0] + boundaries_lines_b[i] - point[1] >= 0]
                                        # Assing points that lie on the boundary line to the bottom cluster
    
    if random_number == 1: # (Upper cluster)
        for i in range(CH.shape[0]):
            if i == 0: # Points in the first cluster (this cluster doesn't have bottom boundary)
                cluster_points[i] = [point 
                                    for point in X 
                                    if boundaries_lines_m[i]*point[0] + boundaries_lines_b[i] - point[1] > 0]
            elif i == CH.shape[0]-1: # Point in the last cluster (this cluster doesn't have top boundary)
                cluster_points[i] = [point 
                                    for point in X 
                                    if boundaries_lines_m[i-1]*point[0] + boundaries_lines_b[i-1] - point[1] <= 0]
            else:        
                    cluster_points[i] = [point # Points in the middle 
                                        for point in X 
                                        if boundaries_lines_m[i-1]*point[0] + boundaries_lines_b[i-1] - point[1] <= 0
                                        and boundaries_lines_m[i]*point[0] + boundaries_lines_b[i] - point[1] > 0]
                                        # # Assing points that lie on the boundary line to the Upper cluster

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
        

def add_centroid(dist_max_cluster, dist_max_point, dist_max, lower_bound, upper_bound):
    # Initialize new_CH as a zero array with the same shape as dist_max_cluster
    new_CH = np.zeros_like(dist_max_cluster)

    # Right
    if (dist_max_cluster[0] - dist_max_point[0] < 0) and (dist_max_cluster[1] - dist_max_point[1] == 0):
        new_CH[0] = np.round(dist_max_cluster[0] + (dist_max / 2))
        new_CH[1] = dist_max_cluster[1]

    # Left
    elif (dist_max_cluster[0] - dist_max_point[0] > 0) and (dist_max_cluster[1] - dist_max_point[1] == 0):
        new_CH[0] = np.round(dist_max_cluster[0] - (dist_max / 2))
        new_CH[1] = dist_max_cluster[1]

    # Top
    elif (dist_max_cluster[1] - dist_max_point[1] < 0) and (dist_max_cluster[0] - dist_max_point[0] == 0):
        new_CH[0] = dist_max_cluster[0]
        new_CH[1] = np.round(dist_max_cluster[1] + (dist_max / 2))

    # Bottom
    elif (dist_max_cluster[1] - dist_max_point[1] > 0) and (dist_max_cluster[0] - dist_max_point[0] == 0):
        new_CH[0] = dist_max_cluster[0]
        new_CH[1] = np.round(dist_max_cluster[1] - (dist_max / 2))

    # Top Right
    elif (dist_max_cluster[0] - dist_max_point[0] < 0) and (dist_max_cluster[1] - dist_max_point[1] < 0):
        new_CH[0] = np.round(dist_max_cluster[0] + (dist_max / 2))
        new_CH[1] = np.round(dist_max_cluster[1] + (dist_max / 2))

    # Top Left
    elif (dist_max_cluster[0] - dist_max_point[0] > 0) and (dist_max_cluster[1] - dist_max_point[1] < 0):
        new_CH[0] = np.round(dist_max_cluster[0] - (dist_max / 2))
        new_CH[1] = np.round(dist_max_cluster[1] + (dist_max / 2))

    # Bottom Right
    elif (dist_max_cluster[0] - dist_max_point[0] < 0) and (dist_max_cluster[1] - dist_max_point[1] > 0):
        new_CH[0] = np.round(dist_max_cluster[0] + (dist_max / 2))
        new_CH[1] = np.round(dist_max_cluster[1] - (dist_max / 2))

    # Bottom Left
    elif (dist_max_cluster[0] - dist_max_point[0] > 0) and (dist_max_cluster[1] - dist_max_point[1] > 0):
        new_CH[0] = np.round(dist_max_cluster[0] - (dist_max / 2))
        new_CH[1] = np.round(dist_max_cluster[1] - (dist_max / 2))

    # After calculating new_CH, ensure it is within the allowed boundaries
    # Check for upper boundary
    if new_CH[0] > upper_bound:
        new_CH[0] = upper_bound
    if new_CH[1] > upper_bound:
        new_CH[1] = upper_bound

    # Check for lower boundary
    if new_CH[0] < lower_bound:
        new_CH[0] = lower_bound
    if new_CH[1] < lower_bound:
        new_CH[1] = lower_bound

    print(f'newCH = {new_CH}, last_CH = {dist_max_cluster}')

    return new_CH


# Calculate the euclidean distance between 2 points
def calculate_euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Calculate the distance from the cluster to the the boundary line (perpendicular distance)
# def calculate_distance_to_line(centroid, m, b):
#     # Distance from point to line (Ax + By + C = 0) is |Ax + By + C| / sqrt(A^2 + B^2)
#     # For y = mx + b, A = -m, B = 1, and C = -b
#     return abs(-m * centroid[0] + centroid[1] + b) / np.sqrt(m**2 + 1)

def check_distance_clusters(CH, cluster_points, boundaries_points):
    # Step 1: Calculate farthest intra-cluster distances
    farthest_distances = {}
    for cluster_id, points in cluster_points.items():
        centroid = CH[cluster_id]
        distances = [calculate_euclidean_distance(centroid, point) for point in points]
        farthest_distance = max(distances)
        farthest_distances[cluster_id] = farthest_distance


     # Step 2: Calculate distances between adjacent cluster centroids
    centroid_distances = {}
    for cluster_id in range(len(CH)):
        if cluster_id == 0:
            # For the first cluster, only calculate distance to the next centroid
            next_centroid = CH[cluster_id + 1]
            centroid_distances[cluster_id] = calculate_euclidean_distance(CH[cluster_id], next_centroid)
        elif cluster_id == len(CH) - 1:
            # For the last cluster, only calculate distance to the previous centroid
            prev_centroid = CH[cluster_id - 1]
            centroid_distances[cluster_id] = calculate_euclidean_distance(CH[cluster_id], prev_centroid)
        else:
            # For other clusters, calculate distance to both the previous and next centroids and take the minimum
            prev_centroid = CH[cluster_id - 1]
            next_centroid = CH[cluster_id + 1]
            distance_to_prev = calculate_euclidean_distance(CH[cluster_id], prev_centroid)
            distance_to_next = calculate_euclidean_distance(CH[cluster_id], next_centroid)
            centroid_distances[cluster_id] = min(distance_to_prev, distance_to_next)


    # Step 3: Compare the farthest intra-cluster distances to the centroid-to-adjacent-centroid distances
    for cluster_id in farthest_distances:
        print(f'farest point distance from cluster {cluster_id}: {farthest_distances[cluster_id]}')
        print(f'mimimum distance between clusters for cluster {cluster_id}: {centroid_distances[cluster_id]}')
        # If the farthest intra-cluster distance is greater than the distance to the closest adjacent centroid
        if farthest_distances[cluster_id] > (0.7 * centroid_distances[cluster_id]):
            return True  # Return True if any farthest distance is greater than the closest adjacent centroid distance

    return False  # Return False if no farthest distances are greater than the closest adjacent centroid distances

 