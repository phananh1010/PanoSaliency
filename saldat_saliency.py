from scipy import stats
import numpy as np
from Quaternion import Quat
from pyquaternion import Quaternion
import head_orientation_lib

import timeit

class Fixation:
    _H = 90
    _W = 160
    _DATASET1 = 1
    _DATASET2 = 2
    
    _gaussian_dict = {}
    _vec_map = None
    _dataset_info_dict = {_DATASET1: [head_orientation_lib.extract_direction_dataset1], _DATASET2: [head_orientation_lib.extract_direction_dataset1]}
    
    def __init__(self, var, dataset):
        self._gaussian_dict = {np.around(_d, 1):stats.multivariate_normal.pdf(_d, mean=0, cov=var) for _d in np.arange(0.0, 180, .1  )}
        f_extract_direction = self._dataset_info_dict[dataset][0]
        self._vec_map = self.create_pixel_vecmap(f_extract_direction)
        
    def gaussian_from_distance(self, _d):
        temp = np.around(_d, 1)
        return self._gaussian_dict[temp] if temp in self._gaussian_dict else 0.0
        
    def create_pixel_vecmap(self, f_extract_direction):
        vec_map = np.zeros((self._H, self._W)).tolist()
        for i in range(self._H):
            for j in range(self._W):
                theta, phi = head_orientation_lib.pixel_to_ang(i, j, self._H, self._W)
                t = Quat([0.0, theta, phi]).q #nolonger use Quat
                q = Quaternion([t[3], t[2], -t[1], t[0]])
                vec_map[i][j] = f_extract_direction(q)
        return vec_map
    
    def create_saliency(self, fixation_list, verbal=False):
        idx = 0
        heat_map = np.zeros((self._H, self._W))
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

                idx += 1;
                if verbal == False: continue
                if idx % 10000 == 0:
                      print self._W * self._H, idx, i, j, heat_map[i, j], d, dd_time, gau_time
                if d < 5: 
                      print '<5 degree: ---->', self._W * self._H, idx, i, j, heat_map[i, j], d, dd_time, gau_time
        return heat_map


    
#    def init(_topic, _seek_time, _var, _ratio=1.0/10):
#    gaussian_dict = {np.around(_d, 1):stats.multivariate_normal.pdf(_d, mean=0, cov=_var) for _d in np.arange(0.0, 180, .1  )}
    
#    video_name = topic_dict[_topic]
#    vcap = cv2.VideoCapture(video_name)#roller65.webm, paris.mp4; ocean40.webm; venise.webm
#    vcap.set(cv2.cv2.CAP_PROP_POS_MSEC, _seek_time * 1000)
#    width = vcap.get(cv2.cv2.CAP_PROP_FRAME_WIDTH)   # float
#    height = vcap.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT) # float
#    width = int(width * _ratio)
#    height = int(height * _ratio)

#    res, frame = vcap.read()
#    frameG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    frameS = cv2.resize(frameG, (width, height))
    
#    vec_map = create_pixel_vecmap(height, width)

#    return width, height, frameS, vec_map, gaussian_dict

