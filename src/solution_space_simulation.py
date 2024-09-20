def transform(fitness: float,
              a: float,
              t: float) -> float: 
    '''
    Funciton to update the fitness value according to the difference
    between a threshold value `t` and the the current fitness value.

    Arguments:
    ----------
    fitness: float
        The current fitness value.
    
    a: float
        The selection pressure on `fitness` that is in effect below
        the threshold value `t`. The scaler of the different in
        `t` - `fitness`.

    t: float
        The threshold value, below which a fitness value `fitness` is
        reduced in the updated fitness value. 
    
    Returns:
    --------
    updated_fitness: float
        The updated fitness value once it has passed through the
        transformation.
    '''

    if fitness > t:
        return fitness
    else:
        g_x = a * ((t - fitness))
        f_x = fitness - g_x
        updated_fitness = max(f_x, 0)

        return updated_fitness