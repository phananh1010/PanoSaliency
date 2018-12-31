from scipy import stats
import numpy as np
from Quaternion import Quat
from pyquaternion import Quaternion
import head_orientation_lib

import timeit

class Fixation:
    _DATASET1 = 1
    _DATASET2 = 2
    
    _gaussian_dict = {}
    _vec_map = None
    #_dataset_info_dict = {_DATASET1: [head_orientation_lib.extract_direction_dataset1], _DATASET2: [head_orientation_lib.extract_direction_dataset1]}
    
    def __init__(self, var):
        self._gaussian_dict = {np.around(_d, 1):stats.multivariate_normal.pdf(_d, mean=0, cov=var) for _d in np.arange(0.0, 180, .1  )}
        #self._f_extract_direction = self._dataset_info_dict[dataset][0]
        self._vec_map = self.create_pixel_vecmap()
    
    def f_extract_direction(self, q):
        return head_orientation_lib.extract_direction_dataset1(q)
        
    def gaussian_from_distance(self, _d):
        temp = np.around(_d, 1)
        return self._gaussian_dict[temp] if temp in self._gaussian_dict else 0.0
        
    def create_pixel_vecmap(self):
        vec_map = np.zeros((head_orientation_lib.H, head_orientation_lib.W)).tolist()
        for i in range(head_orientation_lib.H):
            for j in range(head_orientation_lib.W):
                theta, phi = head_orientation_lib.pixel_to_ang(i, j, head_orientation_lib.H, head_orientation_lib.W)
                t = Quat([0.0, theta, phi]).q #nolonger use Quat
                q = Quaternion([t[3], t[2], -t[1], t[0]])
                vec_map[i][j] = self.f_extract_direction(q)
        return vec_map
    
    def create_saliency(self, fixation_list, dataset):
        idx = 0
        heat_map = np.zeros((head_orientation_lib.H, head_orientation_lib.W))
        for i in range(heat_map.shape[0]):
            for j in range(heat_map.shape[1]):
                qxy = self._vec_map[i][j]
                for fixation in fixation_list:
                    q0 = fixation[1] 
                    btime = timeit.default_timer()
                    d = head_orientation_lib.degree_distance(q0, qxy)

                    dd_time = timeit.default_timer() - btime

                    heat_map[i, j] += 1.0 * self.gaussian_from_distance(d)
                    gau_time = timeit.default_timer() - btime - dd_time

#                idx += 1;
#                if verbal == False: continue
#                if idx % 10000 == 0:
#                      print self._W * self._H, idx, i, j, heat_map[i, j], d, dd_time, gau_time
#                if d < 5: 
#                      print '<5 degree: ---->', head_orientation_lib.W * head_orientation_lib.H, idx, i, j, heat_map[i, j], d, dd_time, gau_time
                  
        heat_map1 = np.fliplr(heat_map)
        heat_map1 = np.flipud(heat_map1)
        pos = head_orientation_lib.W/2
        temp = np.copy(heat_map1[:, pos:])
        heat_map1[:, pos:] = heat_map1[:, :pos]
        heat_map1[:, :pos] = temp
        
        if dataset == self._DATASET1:
            heat_map1 = np.fliplr(heat_map1)
            heat_map1 = np.flipud(heat_map1)
            
        elif dataset == self._DATASET3:
            pos = head_orientation_lib.W/4
            npos = head_orientation_lib.W/4*3
            temp = np.copy(heat_map1[:, pos:])
            heat_map1[:, npos:] = heat_map1[:, :pos]
            heat_map1[:, :npos] = temp
        
        return heat_map1

