import os
import numpy as np
from Quaternion import Quat
from pyquaternion import Quaternion

from sklearn import model_selection
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import head_orientation_lib
reload(head_orientation_lib)

class HeadOrientation:
    _DATASET1 = 1
    _DATASET2 = 2
    _DATASET3 = 3
    
    _dirpath_dat1 = ''#u'/home/u9168/salnet/Datasets/Dataset1/results/'
    _dirpath_dat2 = ''#u'/home/u9168/salnet/Datasets/Dataset2/Formated_Data/Experiment_1/'
    _dirpath_dat3 = ''#u'/home/u9168/saliency_dataset/data/head-orientation/dataset3/sensory/orientation'
    _file_ext1 = '.txt'
    _file_ext2 = '.csv'
    _file_ext3 = '.csv'
    _dataset_info_dict = {_DATASET1:[], _DATASET2:[], _DATASET3:[]}
    
    _topic_dict = {_DATASET1:['venise', 'diving', 'roller', 'paris', 'timelapse'], \
              _DATASET2:['1', '0', '3', '2', '5', '4', '7', '6', '8'],\
              _DATASET3:['coaster_', 'coaster2_', 'diving', 'drive', 'game', 'landscape', 'pacman', 'panel', 'ride', 'sport']}
    
    def __init__(self, dir_path1, dir_path2, dir_path3, file_ext1, file_ext2, file_ext3):
        self._dirpath_dat1 = dir_path1
        self._dirpath_dat2 = dir_path2
        self._dirpath_dat3 = dir_path3
        self._file_ext1 = file_ext1
        self._file_ext2 = file_ext2
        self._file_ext3 = file_ext3
        self._dataset_info_dict = {self._DATASET1:[self._dirpath_dat1, self._file_ext1, self.parse_dat1, head_orientation_lib.extract_direction_dataset1], \
                                self._DATASET2:[self._dirpath_dat2, self._file_ext2, self.parse_dat2, head_orientation_lib.extract_direction_dataset2], \
                               self._DATASET3:[self._dirpath_dat3, self._file_ext3, self.parse_dat3, head_orientation_lib.extract_direction_dataset3]}
        
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
    
    def parse_dat3(self, _file_name):#for Wen Lo dataset
        temp = open(_file_name).read().split('\n')[1:-1]#remove header and useless last line
        temp2 = [map(float, item.split(',')) for item in temp]

        for i, _ in enumerate(temp2):
            fid, _, _, _, _, _, _, theta, phi, psi = temp2[i]
            tstamp = fid * 1.0/30
            t = Quat([psi, theta, phi]).q #nolonger use Quat
            q = Quaternion([t[3], t[2], -t[1], t[0]])
            
            w, x, y, z = q.elements#{} + {}i + {}j + {}k, a=w, b=x, c=y, d=z
            temp2[i] = [tstamp, fid, z, y, x, w]

        return np.array(temp2)
    
    def load_filename_list(self, dataset, topic):
        #load all headpos log of all users for a given dataset & video_topic
        filename_list = []
        if dataset != self._DATASET1 and dataset != self._DATASET2 and dataset != self._DATASET3:
            print 'ERROR, dataset number must be either 1 or 2 or 3'
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
    
    def get_fixation(self, vector_ds, time, _bp=3, _ap=1, filter_fix=True):
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
                
        #now filter the fixation before returning
        pixel_set0, fix_list0 = self.create_fixation_pixelset(result)
        if filter_fix == True:
            pixel_set, idx_list = self.filter_fixation(fix_list0)
            return [fix_list0[idx] for idx in idx_list]
        else:
            return fix_list0
    
    def create_fixation_pixellist(self, fixation_list):
        #return list of (hi, wi) coord
        pixel_list = []
        for time, v, _, _ in fixation_list:
            theta, phi = head_orientation_lib.vector_to_ang(v)
            x, y = head_orientation_lib.ang_to_geoxy(theta, phi, head_orientation_lib.H, head_orientation_lib.W)
            pixel_list.append((x, y))
        return pixel_list

    def create_fixation_pixelset(self, fixation_list):
        #return set of (hi, wi) coord, eliminate redundancy
        pixel_set = set()
        orifix_list = []
        for time, v, _, _ in fixation_list:
            theta, phi = head_orientation_lib.vector_to_ang(v)
            x, y = head_orientation_lib.ang_to_geoxy(theta, phi, head_orientation_lib.H, head_orientation_lib.W)
            if (x, y) not in pixel_set:
                pixel_set.add((x, y))
                orifix_list.append([time, v, 0, 0])
        return pixel_set, orifix_list    
    
    def filter_fixation(self, _fix_list):
        result = set()
        _geoxy_set = self.create_fixation_pixellist(_fix_list)

        X = [[xy[0], xy[1]] for xy in _geoxy_set]
        #labels_true += [0 for xy in geoxy_false_set]
        std = StandardScaler()
        X = std.fit_transform(X)
        db = DBSCAN(eps=0.3, min_samples=3)
        db.fit_predict(X)
        temp = std.inverse_transform(X[db.core_sample_indices_])

        return set([(int(item[0]), int(item[1])) for item in temp]), db.core_sample_indices_

    def create_fixation_map(self, fixation_list, dataset):
        result = np.zeros(shape=(head_orientation_lib.H, head_orientation_lib.W), dtype=np.int)
        pixel_list = self.create_fixation_pixellist(fixation_list)
        for x, y in pixel_list:
            result[x, y] = 1
            

        
        if dataset == self._DATASET2:
            result1 = np.fliplr(result)
            result1 = np.flipud(result1)
        elif dataset == self._DATASET1:
            result1 = np.fliplr(result)
            result1 = np.flipud(result1)
            result1 = np.fliplr(result1)
        elif dataset == self._DATASET3:
            pos = head_orientation_lib.W/4
            npos = head_orientation_lib.W/4*3
            
            result1 = np.fliplr(result)
            temp = np.copy(result1[:, pos:])
            result1[:, npos:] = result1[:, :pos]
            result1[:, :npos] = temp
            result1 = np.flipud(result1)

        else:
            print 'INVALID dataset'
            raise
            
        return result1