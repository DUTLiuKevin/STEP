from geopy.distance import distance

def get_total_distance(central_loc_degree: np.ndarray)->np.ndarray:
    """Calculate the storm total travel distance in km 
    :param central_loc_degree: the lat and lon of storm central locations at each time step
    :return: a 1-D array of travel distance for each storm
    """
    storm_distance = np.zeros(central_loc_degree.shape[1])
    for i in range(central_loc_degree.shape[1]):
        locations = central_loc_degree[:,i]

        point1 = 0
        total_dist = 0
        for j in range(len(locations)):
            if np.isscalar(locations[j]):
                continue
            else:
                if np.isscalar(point1):
                    point1 = locations[j]
                else:
                    point2 = locations[j]
                    dist = distance((point1[1],point1[0]),(point2[1],point2[0])).km
                    print(dist)
                    total_dist = total_dist+dist
                    point1 = point2
        storm_distance[i]=total_dist
    return(storm_distance)