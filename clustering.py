import numpy as np

def clusterize(data_in,
               noise_map,
               N_sigma_high=5,
               N_sigma_low=3,
               cluster_dim=(3,3)):
    
    """
    Perform clustering on small pixel detectors, by scanning a cluster "window" across pixels 

    Args:
        data_in (numpy.ndarray): Input data representing the x-ray charge integrating pixel detector.
        noise_map (numpy.ndarray): Noise map.
        N_sigma_high (int): Multiplier of noise above which pixels may be considered as  
                            the center of a cluster. Defaults to 5.
        N_sigma_low (int): Multiplier of noise above which pixels are considered to contain 
                           charge from a photon hit. Defaults to 3.
        cluster_dim (tuple): Dimension of the window used to find clusters. Must be a pair of odd numbers. Defaults to (3,3).

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
    # copy the array so that original array is not modified after calling this function
    data_in = data_in.copy()
 
    # unsqueeze data_in to force shape (trains, cells, pixels_y, pixels_x)
    initial_shape = data_in.shape
    while len(data_in.shape) < 4: 
        data_in = data_in[np.newaxis]
    # unsqueeze noise_map to force shape (cells, pixels_y, pixels_x)
    while len(noise_map.shape) < 3: 
        noise_map = noise_map[np.newaxis]
        
    trains = data_in.shape[0]
    memory_cells = data_in.shape[1]
    pixels_y = data_in.shape[2]
    pixels_x = data_in.shape[3]

    data_clu = data_in.copy()
    clusters = np.zeros(data_in.shape, dtype=np.int8)

    # 0. Loop over trains and memory cells
    for train in range(trains):
        for mc in range(memory_cells):
        
            # 1. Find photon clusters per frame
            photon_hits = np.transpose(np.where(data_in[train, mc]>N_sigma_high*noise_map[mc])).tolist()
            
            for phy, phx in photon_hits:
                
                # discard photon hits on sensor borders
                if ((cluster_dim[0]//2 <= phy < pixels_y-cluster_dim[0]//2) and
                    (cluster_dim[1]//2 <= phx < pixels_x-cluster_dim[1]//2)):
                    
                    # 2. Define cluster window
                    if cluster_dim[0]%2: # odd cluster_dim
                        window_y = np.s_[phy - cluster_dim[0]//2 : phy + cluster_dim[0]//2+1]
                        cluster_center_y = cluster_dim[0]//2
                    else:                # even cluster_dim
                        if data_in[train, mc][phy-1, phx] > data_in[train, mc][phy+1, phx]: # propagate cluster downwards
                            window_y = np.s_[phy - cluster_dim[0]//2 : phy + cluster_dim[0]//2]
                            cluster_center_y = cluster_dim[0]//2
                        else:                                   # propagate cluster upwards
                            window_y = np.s_[phy+1 - cluster_dim[0]//2 : phy + cluster_dim[0]//2+1]
                            cluster_center_y = cluster_dim[0]//2-1

                    if cluster_dim[1]%2: # odd cluster_dim
                        window_x = np.s_[phx - cluster_dim[1]//2 : phx + cluster_dim[1]//2+1]
                        cluster_center_x = cluster_dim[1]//2
                    else:                # even cluster_dim
                        if data_in[train, mc][phy, phx-1] > data_in[train, mc][phy, phx+1]: # propagate cluster leftwards
                            window_x  = np.s_[phx - cluster_dim[1]//2 : phx + cluster_dim[1]//2]
                            cluster_center_x = cluster_dim[1]//2
                        else:                                   # propagate cluster rightwards
                            window_x = np.s_[phx+1 - cluster_dim[1]//2 : phx+1 + cluster_dim[1]//2]
                            cluster_center_x = cluster_dim[1]//2-1

                    window = np.s_[..., window_y, window_x]
                    cluster = data_in[train, mc][window]
                    cluster_noise = noise_map[mc][window]
                    
                    # 3. Check if cluster center is the local maximum
                    if cluster[cluster_center_y, cluster_center_x] == cluster.max():
                        
                        
                        # 4. Pass cluster sum to central pixel, and zero other pixles from cluster
                        cluster_above_noise = np.where(cluster>cluster_noise*N_sigma_low, True, False)
                        data_clu[train, mc][window] = np.where(cluster_above_noise, 0, data_clu[train, mc][window])
                        data_clu[train, mc][phy, phx] = cluster[cluster_above_noise].sum()
                        
                        # 5. Fill clusters array with cluster size identification
                        clusters[train, mc][window] = cluster_above_noise*cluster_above_noise.sum()*-1
                        clusters[train, mc][phy, phx] = cluster_above_noise.sum()
                        
                        # 6. Zero-out already clusterized pixels, so that they are not attributed to other clusters
                        data_in[train, mc][window][cluster_above_noise] = 0
                    
    if len(data_clu.shape) > len(initial_shape):
        data_clu = data_clu.squeeze()
        clusters = clusters.squeeze()

    return data_clu, clusters
