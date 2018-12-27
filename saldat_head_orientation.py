import os
import numpy as np
from pyquaternion import Quaternion

import head_orientation_lib
reload(head_orientation_lib)

class HeadOrientation:
    _DATASET1 = 1
    _DATASET2 = 2
    
    _dirpath_dat1 = ''#u'/home/u9168/salnet/Datasets/Dataset1/results/'
    _dirpath_dat2 = ''#u'/home/u9168/salnet/Datasets/Dataset2/Formated_Data/Experiment_1/'
    _file_ext1 = ''#.txt
    _file_ext2 = ''#.csv
    _dataset_info_dict = {_DATASET1:[], _DATASET2:[]}
    

    
    def __init__(self, dir_path1, dir_path2, file_ext1, file_ext2):
        self._dirpath_dat1 = dir_path1
        self._dirpath_dat2 = dir_path2
        self._dataset_info_dict = {1:[self._dirpath_dat1, self._file_ext1, self.parse_dat1, head_orientation_lib.extract_direction_dataset1], \
                                2:[self._dirpath_dat2, self._file_ext2, self.parse_dat2, head_orientation_lib.extract_direction_dataset2]}
        
    def parse_dat1(self, _file_name):#for X.Corbillon dataset
        temp = open(_file_name).read().split('\n')[:-1]
        temp2 = [map(float, item.split(' ')) for item in temp]
        for i, _ in enumerate(temp2):
            item = temp2[i]
            temp2[i] = [item[0], item[1], item[3], item[5], item[4], item[2]]
        return np.array(temp2)

    def parse_dat2(self, _file_name):#for Wu dataset
        temp = open(_file_name).read().split('\n')[1:-1]#remove header and useless last line
        temp2 = [map(float, item.split(',')[1:]) for item in temp]
        #timestamp_list = [datetime.datetime.strptime(item.split(',')[0], "%Y-%m-%d %H:%M:%S.%f") for item in temp]
        for i, _ in enumerate(temp2):
            item = temp2[i]#timestamp, z, y, x, w, ....
            temp2[i] = [item[0], -1, item[3], item[2], item[1], item[4]]
        return np.array(temp2)
    
    def load_filename_list(self, dataset, topic):
        #load all headpos log of all users for a given dataset & video_topic
        filename_list = []
        if dataset != self._DATASET1 and dataset != self._DATASET2:
            print 'ERROR, dataset number must be either 1 or 2'
            raise Exception
        
        dirpath, file_ext, f_parse, f_extract_orientation = self._dataset_info_dict[dataset]

        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith(file_ext) and file.lower().find(topic) >= 0:
                     filename_list.append((os.path.join(root, file)))          
        return dirpath, filename_list, f_parse, f_extract_orientation
        
    
    def load_series_ds(self, filename_list, f_parse):
        series_ds = []

        for idx, file_name in enumerate(filename_list):
            series = f_parse(file_name);
            series_ds.append(series.tolist())
        series_ds = np.array(series_ds)
        return series_ds
    
    
    def headpos_to_headvec(self, series_ds, f_extract_direction):
        #from raw head quarternion, convert to head direction vector
        vector_ds = []
        for series in series_ds:
            vec = []
            #for item in series:
            for idx in np.arange(0, len(series)):
                item = series[idx]
                q = item[2:6]
                v = f_extract_direction(q)#
                vec.append([item[0], v, 0, 0])#time, cur pos, angular vec, angular acc
            vector_ds.append(vec)
        return vector_ds



    def cutoff_vel_acc(self, vector_ds):
        dd = 7
        stats_ds = []
        for vec in vector_ds:
            stats = []
            for idx in range(len(vec)):
                if idx < 5:
                    continue
                dt = vec[idx][0] - vec[idx-dd][0]
                theta = head_orientation_lib.angle_between(vec[idx][1], vec[idx-dd][1])
                v = theta * 1.0 / dt   
                vec[idx][2] = v

                #if idx == 1:
                #   continue
                dv = vec[idx][2] - vec[idx-dd][2]
                a = dv * 1.0 / dt
                vec[idx][3] = a
                item = [vec[idx][0], vec[idx][1], v, a]
                stats.append(item)
            stats_ds.append(stats)
        result = []
        for vector in stats_ds:
            #removing too fast movement
            remove_idx = set()
            collect_mode = 0 #0 is normal, 1 is begin fast, 2 is begin slow
            #print stats_ds[0]
            for idx, (timestamp, vec, v, a) in enumerate(vector):
                if v > 20 and a > 50:
                    collect_mode = 1;
                if a < -40 and collect_mode == 1:#slowing down
                    collect_mode = 2
                if collect_mode == 2 and a > -50  and v < 20:#slowing down finish
                    collect_mode = 0
                if collect_mode == 1 or collect_mode == 2:
                    remove_idx.add(idx)
            result.append([vector[idx] for idx,_ in enumerate(vector) if idx not in remove_idx])
        return result
    
    def get_fixation(self, vector_ds, time, _bp=3, _ap=1):
        dt = 1.0/30
        series_dt = []
        for vector in vector_ds:
            temp = []
            for item in vector: 
                if item[0] >= time - _bp*dt and item[0] <= time + _ap*dt:
                    temp.append(item)
            series_dt.append(temp)
        #get quaternion from the first elements of each users
        result = []
        for series in series_dt:
            for item in series:
                result.append(item)
        return result