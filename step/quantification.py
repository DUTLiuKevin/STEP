from collections import defaultdict
from math import atan, sqrt
import numpy as np
from scipy.ndimage.measurements import center_of_mass

def quantify(tracked_storms: np.ndarray, precip_data: np.ndarray, lat_data: np.ndarray, long_data: np.ndarray,
             time_interval: float, pixel_size: float) -> tuple:
    """Quantitatively describes individual storms in terms of duration, size, mean intensity, and central location.
    :param tracked_storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param precip_data: the precipitation data corresponding to the tracked storms data, with the same dimensions as
    tracked_storms.
    :param lat_data: the latitude data corresponding to each [y][x] in tracked_storms, given as an array of
    dimensions 1 x Rows x Cols.
    :param long_data: the longitude data corresponding to each [y][x] in tracked_storms, given as an array of
    dimensions 1 x Rows x Cols.
    :param time_interval: the period between temporal 'snapshots', given as a float. The user should interpret the
    duration results in terms of the time unit implied here.
    :param pixel_size: the length/width one grid cell represents in the data. The user should interpret the size and
    average intensity results in terms of the distance unit implied here squared.
    :return: A tuple of size four containing the duration of each storm, as well as its size, intensity,
    and central location at each time step, in this order.
    """

    # find the duration of the storms
    durations = get_duration(tracked_storms, time_interval)

    # find the size of each storm in each time slice
    sizes = get_size(tracked_storms, pixel_size)

    # find the average precipitation amount for each storm in each time slice
    averages = get_average(tracked_storms, precip_data)

    # and find the central location for each storm in each time slice
    # central_locs = get_central_loc(tracked_storms, precip_data, lat_data, long_data)

    return durations, sizes, averages# , central_locs


def get_duration(storms: np.ndarray, time_interval: float) -> np.ndarray:
    """Computes the duration (in the time unit of time_interval) of each storm across all time slices given.
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param time_interval: the period between temporal 'snapshots', given as a float.
    :return: An array of length equal to the number of tracked storms + 1, where the value at [x] corresponds to
    the duration of the storm x. The index 0 (referring to the background) is always 0 and provided for ease of
    indexing.
    """

    # find the number of time slices in the data
    lifetime = storms.shape[0]

    # initialize a new dictionary
    duration_dict = defaultdict(int)

    # and the number of storms
    total_storms = len(np.unique(storms))

    # and an array to store the result in, where the value found at each index corresponds to the duration that storm
    result = np.zeros(total_storms)

    # then, for each time slice
    for time_index in range(lifetime):
        # compute the labels that appear in that time slice
        curr_labels = np.unique(storms[time_index])

        # for each label in the tracked storm data
        for label in range(total_storms):
            # if it appears in the current time slice
            if label and np.isin(label, curr_labels):
                # increment the number of time slices it appears in
                # (and if we haven't seen it before, set it to 1 in the dictionary (this is a property of defaultdict)
                duration_dict[label] += 1

    for key, value in duration_dict.items():
        if key:
            result[key] = value

    result = result * time_interval

    return result


def get_size_prj(storms: np.ndarray, grid_cell_degree: float, lat_data: np.ndarray, lon_data: np.ndarray) -> np.ndarray:
    """
    Compute the size (km^2) of each storm consider degree conversion
    :param stroms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param grid_cell_degree: 0.25 degree for ERA5 storms
    :param lat_data: 经度
    :param lon_data: 纬度
    :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the size of the storm at t=y,
    storm=x. Except in the case of index 0, which is always 0 for any t.
    """
    # 创造一个面积矩阵，每个pixel对应该纬度下, 0.25 * 0.25 degree pixel应该有的面积 km^2
    # 创建meshgrid
    lon_2, lat_2 = np.meshgrid(lon_data, lat_data)
    # 计算每个pixel的面积 (km^2)
    pixel_area = np.cos(lat_2 * np.pi / 180) * 111 * 111 * grid_cell_degree * grid_cell_degree

    # find the number of time slices in the data
    lifetime = storms.shape[0]

    # TODO: CHANGED TO LEN, NOT SURE HOW WORKING BEFORE
    # and the number of storms
    total_storms = len(np.unique(storms))

    # initialize an array with dimensions number of time slices by number of storms
    result = np.zeros((lifetime, total_storms))

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        # for each label that appears in this time slice (that's not the background)
        for label in labels:
            if label:
                # add up its coverage area over the pixel_area_matrix
                storm_size = np.sum(np.where(storms[time_index] == label, pixel_area, 0))

                # and place it at that correct location in the array to return
                result[time_index][label] = storm_size

    return result


def get_size(storms: np.ndarray, grid_cell_size: float) -> np.ndarray:
    """Computes the size (in the distance unit of grid_cell_size) of each storm across all time slices given.
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param grid_cell_size: the area one grid cell represents in the data, given as a float.
    :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the size of the storm at t=y,
    storm=x. Except in the case of index 0, which is always 0 for any t.
    """

    # find the number of time slices in the data
    lifetime = storms.shape[0]

    # TODO: CHANGED TO LEN, NOT SURE HOW WORKING BEFORE
    # and the number of storms
    total_storms = len(np.unique(storms))

    # initialize an array with dimensions number of time slices by number of storms
    result = np.zeros((lifetime, total_storms))

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        # for each label that appears in this time slice (that's not the background)
        for label in labels:
            if label:

                # compute its number of grid cells using a map and reduce technique
                storm_size = np.sum(np.where(storms[time_index] == label, 1, 0))

                # and place it at that correct location in the array to return
                result[time_index][label] = storm_size

    # multiply the number of grid cells in each storm by the grid cell size
    result = result * grid_cell_size

    return result


def get_average(storms: np.ndarray, precip: np.ndarray) -> np.ndarray:
    """Computes the average intensity of each storm across all time slices given.
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param precip: the precipitation data corresponding to the tracked storms, with the same dimensions as
    tracked_storms.
    :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the mean intensity of the
    storm at t=y, storm=x. Except in the case of index 0, which is always 0 for any t.
    """

    # find the number of time slices in the data
    lifetime = storms.shape[0]

    # and the number of storms
    total_storms = len(np.unique(storms))

    # initialize an array with dimensions number of time slices by number of storms
    result = np.zeros((lifetime, total_storms))

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        # for each label that appears in this time slice (that's not the background)
        for label in labels:

            if label:
                # find the precipitation where it appears in the current time slice
                storm_precip = np.where(storms[time_index] == label, precip[time_index], 0)

                # sum the precipitation
                storm_precip_sum = np.sum(storm_precip)

                # find the number of grid cells belonging to the storm
                storm_size = np.sum(np.where(storms[time_index] == label, 1, 0))

                # find the storm's average precipitation in this time slice
                storm_avg = storm_precip_sum / storm_size

                # and store it in the appropriate place in our result array
                result[time_index][label] = storm_avg

    return result


def get_central_loc(storms: np.ndarray, precip: np.ndarray, lat_data: np.ndarray, lon_data: np.ndarray) \
        -> np.ndarray:
    """Computes the central location on the earth's surface of each storm across all time slices given.
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param precip: the precipitation data corresponding to the tracked storms data, with the same dimensions as
    tracked_storms.
    :param lat_data: lat_data. 1 * lenth array
    :param lon_data: lon_data 1 * lenth array
    :param size_array: the array returned by get_size(), a lifetime x total_storms array where the value found at [y][x]
    corresponds to the size of the storm at time=y, storm=x.
    :param lifetime: the number of time slices in the data, given as an integer.
    :param total_storms: the total number of storms INCLUDING the background, given as an integer.
    :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the central location of the
    storm at t=y, storm=x. Except in the case of index 0, which is always 0 for any t.
    """

    # create mesh grid of lat and lon data
    lons, lats = np.meshgrid(lon_data, lat_data)

    lifetime = storms.shape[0]

    total_storms = len(np.unique(storms))

    # initialize an array to store our result, but of type object to allow us to store an array in each cell
    result = np.zeros((lifetime, total_storms)).astype(object)

    # create arrays of x, y, and z values for the cartesian grid in R3
    # np.cos(rad) not degree
    x_array = np.cos(lats * np.pi / 180) * np.cos(lons * np.pi / 180)
    y_array = np.cos(lats * np.pi / 180) * np.sin(lons * np.pi / 180)
    z_array = np.sin(lats * np.pi / 180)

    # create an array to hold each central location as we calculate it
    central_location = np.empty(2)

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        for label in labels:
            # if the storm exists in this time slice
            if label:

                # find the sum of the precipitation values belonging to the storm
                sum_precipitation = np.sum(np.where(storms[time_index] == label, precip[time_index], 0))

                # and compute the intensity weighted averages
                x_avg = np.sum(np.where(storms[time_index] == label, ((x_array * precip[time_index]) /
                                                                      sum_precipitation), 0))

                y_avg = np.sum(np.where(storms[time_index] == label, ((y_array * precip[time_index]) /
                                                                      sum_precipitation), 0))

                z_avg = np.sum(np.where(storms[time_index] == label, ((z_array * precip[time_index]) /
                                                                      sum_precipitation), 0))

                h_avg = sqrt((x_avg ** 2) + (y_avg ** 2))

                # the central location on earth's surface is given by the following
                central_location[0] = 2 * atan(y_avg / (sqrt((y_avg ** 2) + (x_avg ** 2)) + x_avg))
                central_location[1] = 2 * atan(z_avg / (sqrt((z_avg ** 2) + (h_avg ** 2)) + h_avg))

                # and we place it in the appropriate spot in the array
                result[time_index][label] = central_location

                # reset the central location - this seems to be necessary here
                central_location = np.zeros(2)

    return result


def get_max_intensity(storms: np.ndarray, precip: np.ndarray) -> np.ndarray:
    """Computes the average intensity of each storm across all time slices given.
        :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
        Time x Rows x Cols.
        :param precip: the precipitation data corresponding to the tracked storms, with the same dimensions as
        tracked_storms.
        :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the mean intensity of the
        storm at t=y, storm=x. Except in the case of index 0, which is always 0 for any t.
        """

    # find the number of time slices in the data
    lifetime = storms.shape[0]

    # and the number of storms
    total_storms = len(np.unique(storms))

    # initialize an array with dimensions number of time slices by number of storms
    result = np.zeros((lifetime, total_storms))

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        # for each label that appears in this time slice (that's not the background)
        for label in labels:

            if label:
                # find the precipitation where it appears in the current time slice
                storm_precip = np.where(storms[time_index] == label, precip[time_index], 0)

                # get the maximum precipitation
                storm_precip_max = np.max(storm_precip)

                # find the number of grid cells belonging to the storm
                # storm_size = np.sum(np.where(storms[time_index] == label, 1, 0))

                # find the storm's average precipitation in this time slice
                # storm_avg = storm_precip_sum / storm_size

                # and store it in the appropriate place in our result array
                result[time_index][label] = storm_precip_max

    return result


def get_central_loc_degree(storms: np.ndarray, precip: np.ndarray, lat_data: np.ndarray, lon_data: np.ndarray) \
        -> np.ndarray:
    """计算pixel为单位的storm中心
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param precip: the precipitation data corresponding to the tracked storms data, with the same dimensions as
    tracked_storms.
    :param lat_data: lat_data. 1 * lenth array
    :param lon_data: lon_data 1 * lenth array
    :param size_array: the array returned by get_size(), a lifetime x total_storms array where the value found at [y][x]
    corresponds to the size of the storm at time=y, storm=x.
    :param lifetime: the number of time slices in the data, given as an integer.
    :param total_storms: the total number of storms INCLUDING the background, given as an integer.
    :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the central location of the
    storm at t=y, storm=x. Except in the case of index 0, which is always 0 for any t.
    """

    # create mesh grid of lat and lon data
    lon_array, lat_array = np.meshgrid(lon_data, lat_data)

    lifetime = storms.shape[0]

    total_storms = len(np.unique(storms))

    # initialize an array to store our result, but of type object to allow us to store an array in each cell
    result = np.zeros((lifetime, total_storms)).astype(object)

    # create arrays of x, y, and z values for the cartesian grid in R3

    # create an array to hold each central location as we calculate it
    central_location = np.empty(2)

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        for label in labels:
            # if the storm exists in this time slice
            if label:

                # find the sum of the precipitation values belonging to the storm
                sum_precipitation = np.sum(np.where(storms[time_index] == label, precip[time_index], 0))

                # and its intensity weighted centroid
                x_avg = np.sum(np.where(storms[time_index] == label, ((lon_array * precip[time_index]) /
                                                                 sum_precipitation), 0))

                y_avg = np.sum(np.where(storms[time_index] == label, ((lat_array * precip[time_index]) /
                                                                 sum_precipitation), 0))


                # get the corresponding lat and lon data
                central_location[0] = x_avg
                central_location[1] = y_avg

                # and we place it in the appropriate spot in the array
                result[time_index][label] = central_location

                # reset the central location - this seems to be necessary here
                central_location = np.zeros(2)

    return result