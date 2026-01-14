from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial.distance import cdist

def avg_med_distance(meas_pcloud, template):
    # compute the Euclidean distance between the two point clouds
    distances = cdist(meas_pcloud, template, 'sqeuclidean')
    # distances = eucl_opt(meas_pcloud,template)
    # assign each point in cloud1 to its closest point in cloud2
    closest_dist_scan_to_temp = np.min(distances, axis = 1)
    closest_dist_temp_to_scan = np.min(distances, axis = 0)
    loss = np.median(closest_dist_scan_to_temp) + np.median(closest_dist_temp_to_scan)

    return loss

def avg_med_distance_only_temp_to_scan(meas_pcloud, template):
    # compute the Euclidean distance between the two point clouds
    distances = cdist(meas_pcloud, template, 'sqeuclidean')
    # assign each point in cloud1 to its closest point in cloud2
    closest_dist_temp_to_scan = np.min(distances, axis = 0)
    loss = np.median(closest_dist_temp_to_scan)

    return loss


def avg_trim_distance(meas_pcloud, template, trim_per):
    # compute the Euclidean distance between the two point clouds
    distances = cdist(meas_pcloud, template, 'sqeuclidean')
    # distances = eucl_opt(meas_pcloud,template)
    # assign each point in cloud1 to its closest point in cloud2
    closest_dist_scan_to_temp = np.min(distances, axis=1)
    closest_dist_temp_to_scan = np.min(distances, axis=0)
    loss = custom_trim_mean(closest_dist_scan_to_temp, trim_per) + custom_trim_mean(closest_dist_temp_to_scan, trim_per)

    return loss

def custom_trim_mean(input,proportion_to_cut_high):
    data_sorted = np.sort(input)
    n_to_use = int((1.-proportion_to_cut_high)*len(data_sorted))
    mean = np.mean(data_sorted[:n_to_use])
    return mean

