# Code to match storm in previous storms
# @author: xiaoye
# 2021/09/06


def previous_strom_match(result_data, precip_data, max_size, best_matched_storm, time_index, back_step, current_label, curr_size, curr_precip_data, curr_centroid, phi, tau, km):
    max_ratio = 0
    prev_size = 0
    # get previous storms and prcp data
    previous_storms = np.unique(result_data[time_index - back_step])
    prev_precip_data = precip_data[time_index - back_step]
    # compute overlap area, find the largest overlapping ratio, if the ratio > 0.3, direct match,
    # else we calculate similarities
    # k = 0
    for storm in previous_storms:
        if storm == 0:
            continue
        # compute the overlapping area
        # find where the storm exists in the appropriate time slice
        previous_storm = np.where(result_data[time_index - back_step] == storm, 1, 0)
        prev_size = np.sum(previous_storm)

        # selected the overlap area of current storm to prev storm
        overlap_curr_to_prev = np.where(previous_storm == 1, current_label, 0)
        # compute overlapping size
        overlap_size_curr_to_prev = np.sum(overlap_curr_to_prev)
        overlap_ratio_curr_to_prev = overlap_size_curr_to_prev / curr_size

        # selected the overlap area of prev to curr
        overlap_prev_to_curr = np.where(current_label == 1, previous_storm, 0)
        overlap_size_prev_to_curr = np.sum(overlap_prev_to_curr)
        overlap_ratio_prev_to_curr = overlap_size_prev_to_curr / prev_size

        integrated_ratio = overlap_ratio_curr_to_prev + overlap_ratio_prev_to_curr

        # record the output data
        # single_run.loc[k, 'prev_id'] = storm
        # single_run.loc[k, 'prev_size'] = prev_size
        # single_run.loc[k, 'overlap_ratio'] = integrated_ratio
        # add the record to dataframes
        # single_run.to_csv(r"E:\Atmosphere\STEP_results\tracking_results\1H\grow_track_r_4_tr_0.5_lowtr_0.03_tau_0.10.05_phi_0.002_km_15\Single_record_output" +\
        #                  "\\" + str(label) + "_"+ str(storm) + ".csv")

        # find the largest overlapping ratio
        if integrated_ratio > max_ratio:
            max_ratio = integrated_ratio
            temp_matched_storm = storm

        # k = k + 1

    if max_ratio > ratio_threshold:
        best_matched_storm = temp_matched_storm
        max_size = prev_size

    else:

        # then for every labeled storm in the previous time index
        # k = 0
        for storm in previous_storms:

            if storm == 0:  # 如果是背景0 就跳过0并继续
                continue

            # find where the storm exists in the appropriate time slice
            previous_storm = np.where(result_data[time_index - back_step] == storm, 1, 0)

            # compute the size of the previous storm
            prev_size = np.sum(previous_storm)

            # if test:
            #     print('Compare storm {0} in previous time slice with size {1}'.format(storm, prev_size))

            # if test:
            #     print(f'Possible match size: {prev_size}')

            # if the storm is not the background and the size of this storm is greater than that of the previous
            # best match
            if storm and prev_size > max_size:

                similarity_metric = similarity(current_label, previous_storm, curr_precip_data,
                                               prev_precip_data, phi)

                # single_run.loc[k, 'similarity_ratio'] = similarity_metric

                # record similarity record
                # single_run['similarity'] = similarity_metric
                # if their similarity measure is greater than the set tau threshold
                if similarity_metric > tau:

                    # find the precipitation data for this storm
                    prev_storm_precip = np.where(result_data[time_index - back_step] == storm, prev_precip_data, 0)

                    # and its intensity-weighted centroid
                    prev_centroid = center_of_mass(prev_storm_precip)

                    curr_prev_displacement = displacement(curr_centroid, prev_centroid)
                    curr_prev_magnitude = magnitude(curr_prev_displacement)
                    # single_run.loc[k, 'distance'] = curr_prev_magnitude
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

            # k = k + 1
    return max_size, best_matched_storm
