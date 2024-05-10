import numpy as np

def clusterize(data_in,
               noise_map,
               N_sigma_high=5,
               N_sigma_low=3,
               allow_2D=True,
               allow_diag=True,
               E_threshold=0,
               cluster_dim=(3,3),
               direction='vertical'):
    
    """
    Perform clustering algorithm on "strixel" detectors, i.e., detectors with rectangular pixels 
    where charge sharing is excepted to be much more prevalent in one of the dimensions.

    Args:
        data_in (numpy.ndarray): Input data representing the x-ray charge integrating pixel detector.
        noise_map (numpy.ndarray): Noise map.
        N_sigma_high (int): Multiplier of noise above which pixels may be considered as  
                            the center of a cluster. Defaults to 5.
        N_sigma_low (int): Multiplier of noise above which pixels are considered to contain 
                           charge from a photon hit. Defaults to 3.
        allow_2D (bool): Flag to allow 2D clustering. If True, the highest neighbour of the central pixel 
                         on the non-dominant charge sharing direction may be added to the cluster. Defaults to False.
        allow_diag (bool): Flag to allow clustering of diagonal pixels. If True, the highest vertical neighbour of the
                           previously determined horizontal neighbour may be added to the cluster. Defaults to False  
        E_threshold (float): Energy threshold below which found clusters are discarded. Defaults to 0.
        cluster_dim (tuple): Dimension of the window used to find clusters. Must be a pair of odd numbers. Defaults to (3,3).
        direction (str): Dominant charge sharing direction, either 'horizontal' or 'vertical'. Defaults to 'vertical'.

    Returns:
        clu (numpy.ndarray): Clusterized data where the total charge from a cluster is passed to the central pixel.
        cluster_size (numpy.ndarray): Array of the same shape as 'clu', containing the number of pixels belonging to 
                                      the cluster. Center of cluster is marked as positive, and neighbours that belong
                                      to the same cluster is marked as negative:                              
                                      0: pixels that do not belong to any clusters
                                      1: center pixel of a 1 pixel size cluster (single pixel events)
                                      2: center pixel of a 2 pixel size cluster
                                      3: center pixel of a 3 pixel size cluster
                                      ...
                                      -2: neighbour pixels of a 2 pixel size cluster
                                      -3: neighbour pixels of a 3 pixel size cluster
                                      ...
    """
    
    # unsqueeze data_in to force shape (trains, cells, pixels_y, pixels_x)
    initial_shape = data_in.shape
    while len(data_in.shape) < 4: 
        data_in = data_in[np.newaxis]
    # unsqueeze noise_map to force shape (cells, pixels_y, pixels_x)
    while len(noise_map.shape) < 3: 
        noise_map = noise_map[np.newaxis]
        
    # swap y and x axes if charge sharing direction is horizontal
    if direction.lower()=='horizontal':
        data_in = data_in.swapaxes(-1,-2)
        noise_map = noise_map.swapaxes(-1,-2)

    trains = data_in.shape[0]
    memory_cells = data_in.shape[1]
    pixels_y = data_in.shape[2]
    pixels_x = data_in.shape[3]
    
    center_idx = (cluster_dim[0]*cluster_dim[1])//2
    center_y, center_x = cluster_dim[0]//2, cluster_dim[1]//2

    cross_mask = np.zeros(cluster_dim)
    cross_mask[center_y,:] = 1
    cross_mask[:,center_x] = 1
    
    clu = data_in.copy()
    cluster_size = np.zeros(data_in.shape, dtype=np.int8)

    # 0. Loop over trains and memory cells
    for train in range(trains):
        for mc in range(memory_cells):
        
            # 1. Find photon clusters per frame
            photon_hits = np.transpose(np.where(data_in[train, mc]>N_sigma_high*noise_map[mc])).tolist()
            
            for phy, phx in photon_hits:
                # discard photon hits on sensor borders
                if (center_y-1 < phy < pixels_y-center_y) and (center_x-1 < phx < pixels_x-center_x):
                    
                    cluster = data_in[train, mc][phy-center_y:phy+center_y+1, phx-center_x:phx+center_x+1]
                    cluster_noise = noise_map[mc][phy-center_y:phy+center_y+1, phx-center_x:phx+center_x+1]
                    integrated_noise = 0

                    # 2. Check if index is center of cluster (max of direct horizontal/vertical neighbours)
                    if (cluster*cross_mask).argmax() == center_idx:

                        # 3. Sum pixels in cluster
                        # start with cluster center
                        cluster_sum = cluster[center_y,center_x]
                        integrated_noise += np.sqrt(cluster_noise[center_y,center_x])
                        
                        # look for vertical neighbours above N_low_sigma * noise
                        neighbours_down, neighbours_up, neighbours_left, neighbours_right = 0, 0, 0, 0
                        neighbours_left_down, neighbours_left_up, neighbours_right_down, neighbours_right_up = 0,0,0,0

                        # while vertical neighbours downwards are above N_low_sigma, sum them
                        iy = center_y-1
                        while (iy > -1) and (cluster[iy,center_x] > cluster_noise[iy,center_x]*N_sigma_low):
                            cluster_sum += cluster[iy,center_x]
                            integrated_noise += np.sqrt(cluster_noise[iy,center_x])
                            neighbours_down += 1
                            iy -= 1
                            
                        # while vertical neighbours upwards are above N_low_sigma, sum them
                        iy = center_y+1
                        while (iy < cluster_dim[0]) and (cluster[iy,center_x] > cluster_noise[iy,center_x]*N_sigma_low):
                            cluster_sum += cluster[iy,center_x]
                            integrated_noise += np.sqrt(cluster_noise[iy,center_x])
                            neighbours_up += 1
                            iy += 1

                        # optionally, check if there are horizontal neighbours above N_low_sigma * noise. 
                        # If yes sum the highest one, except if it belongs to another horizontal cluster.
                        if allow_2D:
                            # left neighbour
                            if (cluster[center_y,center_x-1]>cluster[center_y,center_x+1] and
                                cluster[center_y,center_x-1]>cluster_noise[center_y,center_x-1]*N_sigma_low and
                                np.argmax(cluster[center_y-1:center_y+2, center_x-1])==1):
                                cluster_sum += cluster[center_y,center_x-1]
                                integrated_noise += np.sqrt(cluster_noise[center_y,center_x-1])
                                neighbours_left = 1
                                
                                if allow_diag:
                                    if (neighbours_down > neighbours_up and # make sure 4 px events are square shaped
                                        cluster[center_y-1,center_x-1]<cluster[center_y,center_x-1] and # TODO: min of all 2x2 
                                        cluster[center_y-1,center_x-1]>cluster_noise[center_y-1,center_x-1]*N_sigma_low): 
                                        cluster_sum += cluster[center_y-1,center_x-1]
                                        integrated_noise += np.sqrt(cluster_noise[center_y-1,center_x-1])
                                        neighbours_left_down = 1
                                    elif (neighbours_up > neighbours_down and
                                          cluster[center_y+1,center_x-1]<cluster[center_y,center_x-1] and
                                          cluster[center_y+1,center_x-1]>cluster_noise[center_y+1,center_x-1]*N_sigma_low):
                                        integrated_noise += np.sqrt(cluster[center_y+1,center_x-1])
                                        neighbours_left_up = 1

                            # right neighbour
                            elif (cluster[center_y,center_x+1]>cluster[center_y,center_x-1] and
                                  cluster[center_y,center_x+1]>cluster_noise[center_y,center_x+1]*N_sigma_low and
                                  np.argmax(cluster[center_y-1:center_y+2, center_x+1])==1):
                                cluster_sum += cluster[center_y,center_x+1]
                                integrated_noise += np.sqrt(cluster_noise[center_y,center_x+1])
                                neighbours_right = 1
                                
                                if allow_diag:
                                    if (neighbours_down > neighbours_up and 
                                        cluster[center_y-1,center_x+1]<cluster_noise[center_y,center_x+1] and
                                        cluster[center_y-1,center_x+1]>cluster_noise[center_y-1,center_x+1]*N_sigma_low):
                                        cluster_sum += cluster[center_y-1,center_x+1]
                                        integrated_noise += np.sqrt(cluster_noise[center_y-1,center_x+1])
                                        neighbours_right_down = 1
                                    elif (neighbours_up > neighbours_down and
                                          cluster[center_y+1,center_x+1]<cluster[center_y,center_x+1] and
                                          cluster[center_y+1,center_x+1]>cluster_noise[center_y+1,center_x+1]*N_sigma_low):
                                        cluster_sum += cluster[center_y+1,center_x+1]
                                        integrated_noise += np.sqrt(cluster_noise[center_y+1,center_x+1])
                                        neighbours_right_up = 1
                                        
                        # 4. Pass cluster sum to center pixel
                        if cluster_sum > E_threshold:
                            
                            # Calculate cluster size (number of pixels in cluster)
                            neighbours = 1+neighbours_down+neighbours_up+neighbours_left+neighbours_right
                            neighbours += neighbours_left_down+neighbours_left_up+neighbours_right_down+neighbours_right_up

                            # Fill cluster_size array
                            cluster_size[train, mc, phy-neighbours_down:phy, phx] = -neighbours # vertical down
                            cluster_size[train, mc, phy:phy+neighbours_up+1, phx] = -neighbours # vertical up
                            cluster_size[train, mc, phy, phx-neighbours_left] = -neighbours     # horizontal left
                            cluster_size[train, mc, phy, phx+neighbours_right] = -neighbours    # horizontal right
                            cluster_size[train, mc, phy+neighbours_right_up, phx+neighbours_right] = -neighbours # Up Right
                            cluster_size[train, mc, phy-neighbours_right_down, phx+neighbours_right_down] = -neighbours # DR
                            cluster_size[train, mc, phy-neighbours_left_down, phx-neighbours_left_down] = -neighbours # DL
                            cluster_size[train, mc, phy+neighbours_left_up, phx-neighbours_left_up] = -neighbours # UL
                            cluster_size[train, mc, phy, phx] = neighbours
                            
                            # Fill clu array
                            clu[train, mc,
                                phy-neighbours_down:phy+neighbours_up+1,
                                phx-neighbours_left:phx+neighbours_right+1] = 0 # set neighbours to zero 
                            
                            # Attribute total energy to central pixel
                            clu[train, mc, phy, phx] = cluster_sum - (integrated_noise/neighbours)*(np.sqrt(neighbours)-1)
                            
                            # Zero-out already clusterized pixels
                            data_in[train, mc, phy-neighbours_down:phy, phx] = 0 # vertical down
                            data_in[train, mc, phy:phy+neighbours_up+1, phx] = 0 # vertical up
                            data_in[train, mc, phy, phx-neighbours_left] = 0     # horizontal left
                            data_in[train, mc, phy, phx+neighbours_right] = 0    # horizontal right
                            data_in[train, mc, phy+neighbours_right_up, phx+neighbours_right] = 0 # Up Right
                            data_in[train, mc, phy-neighbours_right_down, phx+neighbours_right_down] = 0 # D R
                            data_in[train, mc, phy-neighbours_left_down, phx-neighbours_left_down] = 0 # D L
                            data_in[train, mc, phy+neighbours_left_up, phx-neighbours_left_up] = 0 # U L
                            data_in[train, mc, phy, phx] = 0                                

    # restore data_in shape if it was changed 
    if direction.lower()=='horizontal':
        clu = clu.swapaxes(-1,-2)
        cluster_size = cluster_size.swapaxes(-1,-2)
    if len(clu.shape) > len(initial_shape):
        clu = clu.squeeze()
        cluster_size = cluster_size.squeeze()

    return clu, cluster_size

# TODOs: 
# [x] set data_in to zero after being added to cluster to prevent being added to more than 1 cluster
# [ ] Only assumes cluster size of (3, 3) - line 86: cluster = data_in[train, mc][phy-center_y:phy+center_y+1, phx-center_x:phx+center_x+1]
