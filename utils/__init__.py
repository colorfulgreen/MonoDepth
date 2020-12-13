def disp_to_depth(disp, min_depth=0.1, max_depth=100):
    '''Convert network's sigmoid output into depth predition'''
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth
