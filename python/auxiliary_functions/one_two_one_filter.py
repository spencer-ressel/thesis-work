def one_two_one_filter(signal, n_smooths, dim):
    '''
    This is a 1-2-1 filter used to smooth 2D data. The data must be a 2D array
    with time as the first dimension and space as the second dimension. 

    Parameters
    ----------
    signal : numpy.ndarray
        The data to be smoothed.
    n_smooths : int
        The number of times to perform the smoothing.
    dim : string
        The dimension in which to smooth, based on the axes of the input array.

    Returns
    -------
    signal : numpy.ndarray
        The smoothed data.

    '''
    import numpy as np
    
    # Perform the smoothing k times
    for k in range(0,n_smooths):  
        
        # Time dimension length
        segment_length = np.shape(signal)[0]
        
        # Space dimension length
        n_lon = np.shape(signal)[1]                                
        
        # Loop over the time axis of the data and perform the smoothing
        if dim == 'time':
            for i in range(1,segment_length-1):
                signal[i,:] = 1/4*signal[i-1,:] +\
                                1/2*signal[i,:]   +\
                                1/4*signal[i+1,:] 
        
        # Loop over the space axis of the data and perform the smoothing
        elif dim == 'space':
            for i in range(1,n_lon-1):
                signal[:,i] = 1/4*signal[:,i-1] +\
                                1/2*signal[:,i]   +\
                                1/4*signal[:,i+1]  
                                  
    return signal