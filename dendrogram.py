import os
import numpy as np
import math
from scipy import optimize
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

COORD_START = 0 #[X, Y, Z]
COORD_END = 2
RADIUS = 3
TIME = 4

# Extracts clusters from csv file in directory. Returns positions and radius of clustering regions sorted by time
def extract_clusters_csv(dir):
    os.chdir(dir)
    clusters = np.empty((0,5), float)
    num_clusters = []
    time_clusters = []
    for file in os.listdir(dir):
        if file.endswith("_clusters.csv"):
            # I need to normalize the way that im naming the folders
            time = int(file.split('_')[-2])
            file_contents = np.loadtxt(file, delimiter=',', ndmin=2)
            length = file_contents.shape[0]
            num_clusters.append(length)
            time_clusters.append(time)
            coords = np.hstack((file_contents, time * np.ones(length).reshape([length, 1])))
            clusters = np.vstack([clusters, coords])
    plot_num_clusters(num_clusters, time_clusters)
    return clusters[clusters[:, 4].argsort()]

def plot_num_clusters(num, time):
    plt.scatter(time, num)
    plt.xlabel('Time'); plt.ylabel('Number of HEK Clusters');
    plt.show()

# Given two sets of temporal cluster information, calculate IOU for each cluster box. Returns a matrix of IOUs
# between clusters
def dynamic_IOU(t1,t2):
    print(t1.shape[0], t2.shape[0])
    IOU = np.zeros([t1.shape[0], t2.shape[0]])
    for i in range(t1.shape[0]):
        for j in range(t2.shape[0]):
            IOU[i][j] = cluster_IOU(t1[:][i], t2[:][j])
    return IOU

# Given two clusters, calculate IOU.
def cluster_IOU(c1, c2):
    rad_1 = c1[3]; center_1 = c1[0:2]
    rad_2 = c2[3]; center_2 = c2[0:2]
    #find distance between circles:
    vec = center_1 - center_2
    dist = np.linalg.norm(vec)
    if dist > (rad_1 + rad_2):
        # circles are not overlapping
        return 0
    elif dist < abs(rad_1 - rad_2):
        # circles are completely overlapping
        return 1
    else:
        areas = np.pi * (rad_1 ** 2 + rad_2 ** 2)
        intersect_1 = rad_1 ** 2 * math.acos((dist ** 2 + rad_1 ** 2 - rad_2 ** 2) / (2 * dist * rad_1))
        intersect_2 = rad_2 ** 2 * math.acos((dist ** 2 + rad_2 ** 2 - rad_1 ** 2) / (2 * dist * rad_2))
        intersect_3 = np.sqrt((-dist + rad_1 + rad_2) * (dist+rad_1-rad_2) * (dist - rad_1 + rad_2) * (dist + rad_1 + rad_2)) / 2
        intersect = intersect_1 + intersect_2 - intersect_3
        union = areas - intersect
        return intersect/union

# construct dendrogram
def cluster_graph(clusters, cluster_nodes, cluster_edges, num_time, ts):
    first_ts = clusters[0, 4]
    for i in range(int(num_time/ts)+1):
        curr_cluster = clusters[:][clusters[:, 4] == first_ts + i*ts]
        for index in range(curr_cluster.shape[0]):
            cluster_nodes, cluster_edges = add_node(np.array_str(curr_cluster[:][index]), cluster_nodes, cluster_edges)
        if i > 0:
            prev_cluster = clusters[:][clusters[:, 4] == first_ts + (i-1)*ts]
            IOU_matrix = dynamic_IOU(prev_cluster, curr_cluster)
            cluster_edges = identify_agglomeration(IOU_matrix, prev_cluster, curr_cluster, cluster_edges)
    return cluster_nodes, cluster_edges

def identify_agglomeration(IOU_matrix, prev_cluster, curr_cluster, cluster_edges, threshold=0.05):
    cols_to_delete = []
    rows_to_delete = []
    for i in range(IOU_matrix.shape[0]):
        best_index = -1
        best_IOU = 0
        for j in range(IOU_matrix.shape[1]):
            if IOU_matrix[i][j] >= threshold and IOU_matrix[i][j] > best_IOU:
                best_IOU = IOU_matrix[i][j]
                best_index = j
        cluster_edges = add_edge(np.array_str(prev_cluster[:][i]),
                                 np.array_str(curr_cluster[:][best_index]), cluster_edges)

    return cluster_edges

def add_node(node, cluster_nodes, cluster_edges):
    if node not in cluster_nodes:
        cluster_nodes.append(node)
        cluster_edges[node] = []
        return cluster_nodes, cluster_edges
    else:
        print('error')
        return cluster_nodes, cluster_edges

def add_edge(node1, node2, cluster_edges):
    temp = []
    if node1 not in cluster_edges:
        temp.append(node2)
        cluster_edges[node1] = temp
    elif node1 in cluster_edges:
        temp.extend(cluster_edges[node1])
        temp.append(node2)
        cluster_edges[node1] = temp
    else:
        print('error')
    return cluster_edges

def graph(nodes, edges):
    for node in nodes:
        print(node, "-->", [i for i in edges[node]])

def dictionary_to_tuple(cluster_nodes, cluster_edges):
    tuples = []
    for node in cluster_nodes:
        for i in cluster_edges[node]:
            tuples += [[node, i]]
    return tuples

def top_down_dictionary(cluster_edges):
    roots = []
    parent_children = {}
    for source, destination in cluster_edges.items():
        temp = []
        if source not in parent_children:
            parent_children[source] = []
        assert len(destination) <= 1, destination
        if len(destination) < 1:
            roots.append(source)
            continue
        elif destination[0] not in parent_children:
            parent_children[destination[0]] = []
        temp.extend(parent_children[destination[0]])
        temp.append(source)
        parent_children[destination[0]] = temp
    return roots, parent_children

def tree_to_str(file_name, roots, parent_children, max_time, ts):
    with open(file_name, 'w') as trees:
        ids = 0
        string_output = "("
        for root in roots:
            depth = int(max_time/ts)
            output, ids, depth_id = serialize(root, parent_children, ids, depth)
            string_output += f"{output},"
            # trees.write(f"{output},")
        trees.write(string_output[:-1] + ");")

def serialize(tree, parent_children, id, depth):
    if len(parent_children[tree]) < 1:
        id += 1
        # depth -= 1
        return f"t{id-1}:1", id, depth
    else:
        output = '('
        for child in parent_children[tree]:
            child_output, child_id, depth_id = serialize(child, parent_children, id, depth)
            id = child_id
            depth -= 1
            output += child_output + ', '
        output = output[:-2] + f'):{depth}'
        return output, id, depth_id



if __name__ == "__main__":
    clusters = extract_clusters_csv('/Users/andrew/PycharmProjects/tordoff_model/Outputs/11292021_HEK30/11292021_HEK30_values/')
    cluster_edges = {}
    cluster_nodes = []
    cluster_nodes, cluster_edges = cluster_graph(clusters, cluster_nodes, cluster_edges, 150, 5)
    roots, parent_children = top_down_dictionary(cluster_edges)
    # graph(parent_children, parent_children)
    tree_to_str('/Users/andrew/PycharmProjects/tordoff_model/Outputs/11292021_HEK30/dendrogram_clusters_0.txt', roots, parent_children, 150, 5)


