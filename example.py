import os
import numpy as np
import pickle

from matplotlib import pyplot as plt

import head_orientation_lib
import saldat_head_orientation
import saldat_saliency

if __name__ == "__main__":
    #load fixation maps and saliency maps for video named 'venise' at video time t=12.
    topic = 'venise'
    dataset = headoren._DATASET1
    t = 26.
    
    dirpath1 = u'./data/head-orientation/dataset1'
    dirpath2 = u'./data/head-orientation/dataset2/Experiment_1'
    ext1 = '.txt'
    ext2 = '.csv'
    headoren = saldat_head_orientation.HeadOrientation(dirpath1, dirpath2, ext1, ext2)

    
    dirpath, filename_list, f_parse, f_extract_direction = headoren.load_filename_list(dataset, topic)
    series_ds = headoren.load_series_ds(filename_list, f_parse)
    vector_ds = headoren.headpos_to_headvec(series_ds, f_extract_direction)
    vector_ds = headoren.cutoff_vel_acc(vector_ds)
    
    var = 20
    salsal = saldat_saliency.Fixation(var)
    
    fixation_list = headoren.get_fixation(vector_ds, t)
    fmap0 = headoren.create_fixation_map(fixation_list, dataset)

    x, y = [], []
    for i in range(head_orientation_lib.H):
        for j in range(head_orientation_lib.W):
            if fmap0[i,j]> 0: x.append(i); y.append(j)
    plt.imshow(fmap0)
    plt.scatter(y, x, color='red', s=11, edgecolor='None')
    plt.axis([0, 160, 0, 90])
    plt.title('fixation map for dataset: {}, video: {} at time {}'.format(dataset, topic, t))
    plt.figure()

    heat_map0 = salsal.create_saliency(fixation_list, dataset)
    plt.title('saliency map for dataset: {}, video: {} at time {}'.format(dataset, topic, t))
    plt.imshow(heat_map0)