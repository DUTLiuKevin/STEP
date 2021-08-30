# Fix tracking algorithm: 修改similarity计算方法, distance strong threshold = 10 degrees

# annual tracking with different parameters

import copy
import os
from math import sqrt
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import pdist, squareform
from skimage.segmentation import relabel_sequential
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# function to make a folder
def make_folder(save_path):
    # 创建文件夹
    try:
        os.mkdir(save_path)
    except OSError as error:
        print("Folder exists.")


def track(labeled_maps: np.ndarray, precip_data: np.ndarray, monthSeq: np.ndarray, tau_winter: float, tau_summer: float,
          phi: float, km: float, ratio_threshold: float, parent_loc: str) -> np.ndarray:
    """Tracks rainstorm events over time by linking identified storms across consecutive time steps.
    :param monthSeq: 数据的时间轴变量
    :param labeled_maps: the identified storms returned by the identification algorithm, given as an array of
    dimensions Time x Rows x Cols.
    :param precip_data: the raw precipitation data corresponding to the identified storms, with the same dimensions as
    labeled_maps.
    :param tau: the threshold at which a storm is considered similar enough to another to possibly be linked through
    time, given as a float.
    :param phi: the constant to be used in computing similarity between storms, given as a float.
    :param km: the number of grid cells equivalent to 120km in the maps, given as a float.
    :param degree: the degree of vectors of storms displacement at time t and time t-1, given as a float with unit of rad.
    :param parent_loc: location to save output log file
    :return: a Time x Rows x Cols array containing the identified storms, now tracked through time.
    """
    # test = True
    # create a folder of current time
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    # create a folder to save single run information file
    log_folder = os.path.join(parent_loc, dt_string)
    make_folder(log_folder)
    # create a subfolder to save single run output log
    # log_subfolder = log_folder + "//" + "single_run"
    # make_folder(log_subfolder)
    # create a dataframe to save computation data
    log_dataframe = pd.DataFrame()
    # save short report about match
    short_log_dataframe = pd.DataFrame()

    # change monthseq to pandas dataframe
    date = pd.DatetimeIndex(monthSeq)

    global vec_angle
    shape = labeled_maps.shape
    num_time_slices = shape[0]

    # make a copy of the result of the identification algorithm to avoid labeling collisions
    # we will record any labeling changes here
    result_data = copy.deepcopy(labeled_maps)

    # skip labeling t=0, since it is already labeled correctly
    # for every other time slice
    for time_index in range(1, num_time_slices):

        # found the month
        month = date[0].month
        # use different tau for summer and winter
        if month in [4, 5, 6, 7, 8, 9]:
            tau = tau_summer
        else:
            tau = tau_winter

        # find the labels for this time index and the labeled storms in the previous time index
        current_labels = np.unique(labeled_maps[time_index])
        previous_storms = np.unique(result_data[time_index - 1])

        # and prepare the corresponding precipitation data
        curr_precip_data = precip_data[time_index]
        prev_precip_data = precip_data[time_index - 1]

        # determine the maximum label already used to avoid collisions
        # 2021/07/23 之前使用过的最大的label, 避免重复使用
        if time_index == 1:
            max_label_so_far = np.max(result_data[time_index])
            # print("Time {0} Max label is {1}".format(time_index, max_label_so_far))
        else:
            max_label_so_far = np.max(result_data[:time_index])
            # print("Time {0} Max label is {1}".format(time_index, max_label_so_far))
        # print("Time {0}".format(time_index))
        # record time index
        # print("Time slice : {0}".format(time_index + 1))
        # then, for each label in current time index (that isn't the background)
        for label in current_labels:
            if label:

                # for testing only
                # if test:
                # print(f'Current storm label {0}'.format(label))

                # make sure initially the max storm size and best matched storm are 0
                max_size = 0
                best_matched_storm = 0

                # find where the labels of the current storm segment exist in the current time slice
                current_label = np.where(labeled_maps[time_index] == label, 1, 0)
                curr_size = np.sum(current_label)

                # if test:
                #     print(f'Label size: {np.sum(current_label)}')

                # find the precipitation data at those locations
                curr_label_precip = np.where(labeled_maps[time_index] == label, curr_precip_data, 0)


                # and its intensity weighted centroid
                curr_centroid = center_of_mass(curr_label_precip)

                # create a dataframe to store data
                single_run = pd.DataFrame()
                single_run['timeslice'] = [time_index] * (previous_storms.shape[0]-1)
                single_run['curr_id'] = [label] * (previous_storms.shape[0]-1)
                single_run['curr_size'] = [curr_size] * (previous_storms.shape[0]-1)
                single_run['prev_id'] = np.zeros((previous_storms.shape[0]-1))
                single_run['prev_size'] = np.zeros((previous_storms.shape[0]-1))

                single_run['overlap_ratio'] = np.zeros((previous_storms.shape[0]-1))
                single_run['similarity_ratio'] = np.zeros((previous_storms.shape[0]-1))
                single_run['distance'] = np.zeros((previous_storms.shape[0]-1))



                max_ratio = 0
                prev_size = 0
                # compute overlap area, find the largest overlapping ratio, if the ratio > 0.3, direct match,
                # else we calculate similarities
                k = 0
                for storm in previous_storms:
                    if storm == 0:
                        continue
                    # compute the overlapping area
                    # find where the storm exists in the appropriate time slice
                    previous_storm = np.where(result_data[time_index - 1] == storm, 1, 0)
                    prev_size = np.sum(previous_storm)


                    # selected the overlap area of current storm to prev storm
                    overlap_curr_to_prev = np.where(previous_storm == 1, current_label, 0)
                    # compute overlapping size
                    overlap_size_curr_to_prev = np.sum(overlap_curr_to_prev)
                    overlap_ratio_curr_to_prev = overlap_size_curr_to_prev/curr_size

                    # selected the overlap area of prev to curr
                    overlap_prev_to_curr = np.where(current_label == 1, previous_storm, 0)
                    overlap_size_prev_to_curr = np.sum(overlap_prev_to_curr)
                    overlap_ratio_prev_to_curr = overlap_size_prev_to_curr/prev_size

                    integrated_ratio = overlap_ratio_curr_to_prev + overlap_ratio_prev_to_curr

                    # record the output data
                    single_run.loc[k, 'prev_id'] = storm
                    single_run.loc[k, 'prev_size'] = prev_size
                    single_run.loc[k, 'overlap_ratio'] = integrated_ratio
                    # add the record to dataframes
                    # single_run.to_csv(r"E:\Atmosphere\STEP_results\tracking_results\1H\grow_track_r_4_tr_0.5_lowtr_0.03_tau_0.10.05_phi_0.002_km_15\Single_record_output" +\
                    #                  "\\" + str(label) + "_"+ str(storm) + ".csv")


                    # find the largest overlapping ratio
                    if integrated_ratio > max_ratio:
                        max_ratio = integrated_ratio
                        temp_matched_storm = storm

                    k = k + 1

                if max_ratio > ratio_threshold:
                    best_matched_storm = temp_matched_storm
                    max_size = prev_size

                else:

                    # then for every labeled storm in the previous time index
                    k = 0
                    for storm in previous_storms:

                        if storm == 0:  # 如果是背景0 就跳过0并继续
                            continue

                        # find where the storm exists in the appropriate time slice
                        previous_storm = np.where(result_data[time_index - 1] == storm, 1, 0)

                        # compute the size of the previous storm
                        prev_size = np.sum(previous_storm)

                        # if test:
                        #     print('Compare storm {0} in previous time slice with size {1}'.format(storm, prev_size))

                        # if test:
                        #     print(f'Possible match size: {prev_size}')

                        # if the storm is not the background and the size of this storm is greater than that of the previous
                        # best match
                        if storm and prev_size > max_size:

                            similarity_metric  = similarity(current_label, previous_storm, curr_precip_data, prev_precip_data, phi)

                            single_run.loc[k, 'similarity_ratio'] = similarity_metric

                            # record similarity record
                            # single_run['similarity'] = similarity_metric
                            # if their similarity measure is greater than the set tau threshold
                            if similarity_metric > tau:

                                # find the precipitation data for this storm
                                prev_storm_precip = np.where(result_data[time_index - 1] == storm, prev_precip_data, 0)

                                # and its intensity-weighted centroid
                                prev_centroid = center_of_mass(prev_storm_precip)

                                curr_prev_displacement = displacement(curr_centroid, prev_centroid)
                                curr_prev_magnitude = magnitude(curr_prev_displacement)
                                single_run.loc[k, 'distance'] = curr_prev_magnitude
                                # record the distance
                                # single_run['distance'] = curr_prev_magnitude
                                # if the magnitude of their displacement vector is less than 120 km in grid cells
                                if curr_prev_magnitude < km:
                                    # if test:
                                        # print('Distance {0} (pixel) < {1}.'.format(curr_prev_magnitude, km))
                                    # update the best matched storm information
                                    max_size = prev_size
                                    best_matched_storm = storm

                                    # print('Distance {0} (pixel) >= {1}.'.format(curr_prev_magnitude, km))

                        k = k + 1
                # if we found matches
                if max_size:
                    # link the label in the current time slice with the appropriate storm label in the previous
                    result_data[time_index] = np.where(labeled_maps[time_index] == label, best_matched_storm,
                                                       result_data[time_index])
                    single_run['final_id'] = [best_matched_storm] * (previous_storms.shape[0]-1)
                    single_run['match_type'] = ['match'] * (previous_storms.shape[0]-1)

                    # record the ID
                    # single_run['final_id'] = best_matched_storm
                    # single_run['match_type'] = 'Match'
                # otherwise we've detected a new storm
                else:
                    # give the storm a unique label
                    result_data[time_index] = np.where(labeled_maps[time_index] == label, max_label_so_far + 1,
                                                       result_data[time_index])

                    single_run['final_id'] = [max_label_so_far+1] * (previous_storms.shape[0]-1)
                    single_run['match_type'] = ['new'] * (previous_storms.shape[0]-1)

                    # record the ID
                    # single_run['final_id'] = max_label_so_far + 1
                    # single_run['match_type'] = "New"
                    # and increment the known maximum label to make sure we don't reuse a label
                    max_label_so_far += 1

                # save the single run to csv
                # single_run.to_csv(
                #        log_subfolder + "\\" + "timeslice_" + str(time_index) + "_label_" + str(label) + ".csv")
                # append single run data
                log_dataframe = log_dataframe.append(single_run, ignore_index = True)
                # 同时输出一个精简版的record，如果find match则输出match，
                # 如果没找到就输出最小distance record
                if max_size:
                    match_record = single_run[single_run['prev_id'] == best_matched_storm]
                else:
                    # 如果满足tau的门槛，输出distance的最小不为0值
                    if max(single_run['similarity_ratio']) > tau:
                        distance_metric = single_run.loc[single_run['distance'] != 0, 'distance']
                        match_record = single_run[single_run['distance'] == distance_metric.min()]
                        # print(distance_metric.min())
                    else:
                        # 如果没满足tau的门槛，输出最大的similarity
                        match_record = single_run[single_run['similarity_ratio'] == single_run['similarity_ratio'].max()]
                        # print(single_run['similarity_ratio'].values)
                short_log_dataframe = short_log_dataframe.append(match_record, ignore_index = True)

                # if test:
                #     print(f'{label} matched {best_matched_storm} in time slice {time_index + 1}')
                #     print("######################################################################")
    # save the record log dataframe
    # = datetime.now()
    # dt_string = now.strftime("%Y_%m_%d_%H_%M")
    log_file = dt_string + ".csv"
    log_dataframe.to_csv(os.path.join(log_folder, log_file))

    short_log_file = dt_string + "_short.csv"
    short_log_dataframe.to_csv(os.path.join(log_folder, short_log_file))
    # ensure that we've labeled the storms sequentially
    seq_result = relabel_sequential(result_data)[0]
    # 增加一个返回short operaiton record
    return seq_result, short_log_dataframe


def similarity(curr_label_locs: np.ndarray, prev_storm_locs: np.ndarray, curr_raw_data: np.ndarray,
               prev_raw_data: np.ndarray, phi: float) -> float:
    """Computes the similarity measure between two storms.
    :param curr_label_locs: the location of the storm in the current time slice, where 1 marks a cell that is
    part of the storm and is 0 otherwise, given as an array of dimensions Rows x Cols.
    :param prev_storm_locs: the location of the storm in the ~previous~ time slice, with the same type information as
    curr_label_locs.
    :param curr_raw_data: the raw precipitation data corresponding to the current time slice, with the same
    type information as curr_label_locs.
    :param prev_raw_data: the raw precipitation data corresponding to the previous time slice, with the same
    type information as curr_label_locs.
    :param phi: the constant to be used in computing similarity between storms, given as a float.
    :param test: boolean to toggle testing printouts passed on from designation in track().
    :return: the similarity measure of the two storms, as a float.
    """

    # for the similarity computation, it is strongly encouraged to draw out a small example to understand how this is
    # being computed

    # sum the precipitation data for both
    curr_precip_sum = np.sum(np.where(curr_label_locs == 1, curr_raw_data, 0))
    prev_precip_sum = np.sum(np.where(prev_storm_locs == 1, prev_raw_data, 0))

    # find the weight of each cell in both
    # 对两个storm，计算每个storm内pixel的weighted intensity，其他区域设置为0
    curr_weighted_locs = np.where(curr_label_locs == 1, curr_raw_data / curr_precip_sum, 0)
    prev_weighted_locs = np.where(prev_storm_locs == 1, prev_raw_data / prev_precip_sum, 0)

    # turn the label weighting for the current time slice into a 1d array
    # 提取storm A的所有坐标
    curr_coors = np.argwhere(curr_weighted_locs)

    # do the same for the storm weightings in the previous time slice
    # 提取storm B的所有坐标
    prev_coors = np.argwhere(prev_weighted_locs)

    # 计算pairwise euclidean distance
    distance = euclidean_distances(curr_coors, prev_coors, squared=True)

    # 按顺序提取A 和B里的intensity weight
    curr_union_weights = np.zeros(len(curr_coors))
    prev_union_weights = np.zeros(len(prev_coors))

    for index, coors in enumerate(curr_coors):
        curr_union_weights[index] = curr_weighted_locs[coors[0]][coors[1]]

    for index, coors in enumerate(prev_coors):
        prev_union_weights[index] = prev_weighted_locs[coors[0]][coors[1]]

    # 计算weights的X * Y.T
    curr_prev_weights = np.einsum('i, j -> ij', curr_union_weights, prev_union_weights)

    element_wise_similarity = (np.exp(-1 * phi * distance) * curr_prev_weights)

    # the similarity measure is the sum of the all the cells in this new array
    similarity_measure = np.sum(element_wise_similarity)

    # if test:
    #     print(f'Similarity measure: {similarity_measure}')

    return similarity_measure

    """
    # merge the two arrays
    merged_coors = np.concatenate((curr_coors, prev_coors), axis=0)

    # and find their union
    union_helper = [tuple(row) for row in merged_coors]
    union = np.unique(union_helper, axis=0)
    # https://stackoverflow.com/a/31097302

    # place the weights of these locations for each storm in its appropriate time slice (even if they include 0's)
    curr_union_weights = np.zeros(len(union))
    prev_union_weights = np.zeros(len(union))

    for index, coors in enumerate(union):
        curr_union_weights[index] = curr_weighted_locs[coors[0]][coors[1]]
        prev_union_weights[index] = prev_weighted_locs[coors[0]][coors[1]]

    # reshape these arrays to then perform matrix multiplication on them
    curr_union_weights = curr_union_weights.reshape(curr_union_weights.shape[0], 1)
    prev_union_weights = prev_union_weights.reshape(1, prev_union_weights.shape[0])

    # compute their matrix multiplication as this gives the combinations of the desired weights multiplied
    curr_prev_weights = np.einsum('ij, jk -> ik', curr_union_weights, prev_union_weights)
    # https://stackoverflow.com/a/59858877

    # find the corresponding distances to each combination cell in the array of multiplied weights
    union_dists = squareform(pdist(union))

    # and compute the exponential for each multiplied by the corresponding weights previously found
    element_wise_similarity = (np.exp(-1 * phi * union_dists) * curr_prev_weights)

    # the similarity measure is the sum of the all the cells in this new array
    similarity_measure = np.sum(element_wise_similarity)

    if test:
        print(f'Similarity measure: {similarity_measure}')

    return similarity_measure
    """


def displacement(current: np.ndarray, previous: np.ndarray) -> np.array:
    """Computes the displacement vector between the centroids of two storms.
    :param current: the intensity-weighted centroid of the storm in the current time slice, given as a tuple.
    :param previous: the intensity-weighted centroid of the storm in the previous time slice, given as a tuple.
    :return: the displacement vector, as an array.
    """
    return np.array([current[0] - previous[0], current[1] - previous[1]])


def magnitude(vector: np.ndarray) -> float:
    """Computes the magnitude of a vector.
    :param vector: the displacement vector, given as an array.
    :return: its magnitude, as a float.
    """
    return sqrt((vector[0] ** 2) + (vector[1] ** 2))


def angle(vec_one: np.ndarray, vec_two: np.ndarray) -> float:
    """Computes the small angle between two vectors.
    :param vec_one: the first vector, as an array.
    :param vec_two: the second vector, as an array.
    :return: the angle between them in degrees, as a float.
    """
    # for more information on the computation, https://gamedev.stackexchange.com/q/69475 may be helpful
    return np.arccos(np.dot(vec_one, vec_two) / (np.linalg.norm(vec_one) * np.linalg.norm(vec_two)))


def track_backup(labeled_maps: np.ndarray, precip_data: np.ndarray, monthSeq: np.ndarray, tau_winter: float, tau_summer: float,
          phi: float, km: float, test: bool = False) -> np.ndarray:
    """Tracks rainstorm events over time by linking identified storms across consecutive time steps.
    :param monthSeq: 数据的时间轴变量
    :param labeled_maps: the identified storms returned by the identification algorithm, given as an array of
    dimensions Time x Rows x Cols.
    :param precip_data: the raw precipitation data corresponding to the identified storms, with the same dimensions as
    labeled_maps.
    :param tau: the threshold at which a storm is considered similar enough to another to possibly be linked through
    time, given as a float.
    :param phi: the constant to be used in computing similarity between storms, given as a float.
    :param km: the number of grid cells equivalent to 120km in the maps, given as a float.
    :param degree: the degree of vectors of storms displacement at time t and time t-1, given as a float with unit of rad.
    :param test: turn on/off optional testing printouts to help tune parameters, given as a bool with default value
    False.
    :return: a Time x Rows x Cols array containing the identified storms, now tracked through time.
    """
    # change monthseq to pandas dataframe
    date = pd.DatetimeIndex(monthSeq)

    global vec_angle
    shape = labeled_maps.shape
    num_time_slices = shape[0]

    # make a copy of the result of the identification algorithm to avoid labeling collisions
    # we will record any labeling changes here
    result_data = copy.deepcopy(labeled_maps)

    # skip labeling t=0, since it is already labeled correctly
    # for every other time slice
    for time_index in range(1, num_time_slices):
        # found the month
        month = date[0].month
        # use different tau for summer and winter
        if month in [4, 5, 6, 7, 8, 9]:
            tau = tau_summer
        else:
            tau = tau_winter

        # find the labels for this time index and the labeled storms in the previous time index
        current_labels = np.unique(labeled_maps[time_index])
        previous_storms = np.unique(result_data[time_index - 1])

        # and prepare the corresponding precipitation data
        curr_precip_data = precip_data[time_index]
        prev_precip_data = precip_data[time_index - 1]

        # determine the maximum label already seen to avoid collisions
        max_label_so_far = max(np.max(result_data[time_index - 1]), np.max(labeled_maps[time_index]))

        # then, for each label in current time index (that isn't the background)
        for label in current_labels:
            if label:

                # for testing only
                if test:
                    print(f'Label to match: {label} in time slice {time_index + 1}')

                # make sure initially the max storm size and best matched storm are 0
                max_size = 0
                best_matched_storm = 0

                # find where the labels of the current storm segment exist in the current time slice
                current_label = np.where(labeled_maps[time_index] == label, 1, 0)
                curr_size = np.sum(current_label)

                if test:
                    print(f'Label size: {np.sum(current_label)}')

                # find the precipitation data at those locations
                curr_label_precip = np.where(labeled_maps[time_index] == label, curr_precip_data, 0)


                # and its intensity weighted centroid
                curr_centroid = center_of_mass(curr_label_precip)

                max_ratio = 0
                prev_size = 0
                # compute overlap area, find the largest overlapping ratio, if the ratio > 0.3, direct match,
                # else we calculate similarities
                for storm in previous_storms:

                    if storm == 0:
                        continue

                    # compute the overlapping area
                    # find where the storm exists in the appropriate time slice
                    previous_storm = np.where(result_data[time_index - 1] == storm, 1, 0)
                    prev_size = np.sum(previous_storm)
                    # selected the overlap area of current storm to prev storm
                    overlap_curr_to_prev = np.where(previous_storm == 1, current_label, 0)
                    # compute overlapping size
                    overlap_size_curr_to_prev = np.sum(overlap_curr_to_prev)
                    overlap_ratio_curr_to_prev = overlap_size_curr_to_prev/curr_size

                    # selected the overlap area of prev to curr
                    overlap_prev_to_curr = np.where(current_label == 1, previous_storm, 0)
                    overlap_size_prev_to_curr = np.sum(overlap_prev_to_curr)
                    overlap_ratio_prev_to_curr = overlap_size_prev_to_curr/prev_size

                    integrated_ratio = overlap_ratio_curr_to_prev + overlap_ratio_prev_to_curr

                    print("Overlap ratio = {0}".format(integrated_ratio))
                    # find the largest overlapping ratio
                    if integrated_ratio > max_ratio:
                        max_ratio = integrated_ratio
                        temp_matched_storm = storm

                if max_ratio > 0.3:
                    best_matched_storm = temp_matched_storm
                    max_size = prev_size

                else:

                    # then for every labeled storm in the previous time index
                    for storm in previous_storms:

                        if storm == 0:  # 如果是背景0 就跳过0并继续
                            continue

                        # find where the storm exists in the appropriate time slice
                        previous_storm = np.where(result_data[time_index - 1] == storm, 1, 0)

                        # compute the size of the previous storm
                        prev_size = np.sum(previous_storm)

                        if test:
                            print('Compare storm {0} in previous time slice with size {1}'.format(storm, prev_size))

                        # if test:
                        #     print(f'Possible match size: {prev_size}')

                        # if the storm is not the background and the size of this storm is greater than that of the previous
                        # best match
                        if storm and prev_size > max_size:


                            # if their similarity measure is greater than the set tau threshold
                            if similarity(current_label, previous_storm, curr_precip_data, prev_precip_data, phi, test) \
                                    > tau:

                                # find the precipitation data for this storm
                                prev_storm_precip = np.where(result_data[time_index - 1] == storm, prev_precip_data, 0)

                                # and its intensity-weighted centroid
                                prev_centroid = center_of_mass(prev_storm_precip)

                                curr_prev_displacement = displacement(curr_centroid, prev_centroid)
                                curr_prev_magnitude = magnitude(curr_prev_displacement)

                                # if test:
                                # print(f'Current weighted centroid: {curr_centroid}')
                                # print(f'Previous weighted centroid: {prev_centroid}')
                                # print(f'Displacement: {curr_prev_displacement}')
                                # print(f'Distance (pixel): {curr_prev_magnitude}')

                                # if the magnitude of their displacement vector is less than 120 km in grid cells
                                if curr_prev_magnitude < km:
                                    if test:
                                        print('Distance {0} (pixel) < {1}.'.format(curr_prev_magnitude, km))
                                    # update the best matched storm information
                                    max_size = prev_size
                                    best_matched_storm = storm
                                else:
                                    print('Distance {0} (pixel) >= {1}.'.format(curr_prev_magnitude, km))
                                    # otherwise, if the angle between this displacement vector and the previous displacement
                                    # vector associated with that label is less than 120 degrees and it's possible to find
                                    # this angle
                                    # if time_index > 1:
                                    #     predecessor_storms = np.unique(result_data[time_index - 2])
                                    #     if np.isin(storm, predecessor_storms):  # 如果storm在pred storm里
                                    #         # compute the displacement between the possible match and its predecessor
                                    #         predecessor_loc = np.where(result_data[time_index - 2] == storm, 1, 0)
                                    #         pred_precip = np.where(predecessor_loc == 1, precip_data[time_index - 2], 0)
                                    #         pred_centroid = center_of_mass(pred_precip)
                                    #         prev_pred_displacement = displacement(prev_centroid, pred_centroid)
                                    #         vec_angle = angle(curr_prev_displacement, prev_pred_displacement)
                                    #     # 如果不在呢...艹
                                    #     else:
                                    #         vec_angle = 1
                                    # else:
                                    #     vec_angle = 1
                                    # if test:
                                    #     print(f'Angle: {vec_angle}')
                                    # # add the degree threshold in the function input
                                    # if vec_angle < degree:  # equivalent to 120 degree direction difference
                                    #     # update the best matched storm information
                                    #     if test:
                                    #         print('Possible match through angle')
                                    #     max_size = prev_size
                                    #     best_matched_storm = storm

                # if we found matches
                if max_size:
                    # link the label in the current time slice with the appropriate storm label in the previous
                    result_data[time_index] = np.where(labeled_maps[time_index] == label, best_matched_storm,
                                                       result_data[time_index])
                # otherwise we've detected a new storm
                else:
                    # give the storm a unique label
                    result_data[time_index] = np.where(labeled_maps[time_index] == label, max_label_so_far + 1,
                                                       result_data[time_index])
                    # and increment the known maximum label to make sure we don't reuse a label
                    max_label_so_far += 1

                if test:
                    print(f'{label} matched {best_matched_storm} in time slice {time_index + 1}')
                    print("######################################################################")

    # ensure that we've labeled the storms sequentially
    seq_result = relabel_sequential(result_data)[0]

    return seq_result