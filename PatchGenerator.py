from utils import getLabel, getWSI, getRegionFromSlide
import numpy as np
import openslide
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from itertools import cycle

def patch_generator(folder, all_patch_list, 
                    det_patch_list, batch_size=64, 
                    detection_ratio=0.5, levels=[0,1,2],
                    dims=(512,512)):
    '''
    Returns (via yields) the sample image patch and corresponding ground truth mask, in given batch_size, using
    one level in levels list per patch with equal probability
    '''
    
    
    true_batch = int(detection_ratio * batch_size)+1
    all_batch_size = batch_size - true_batch
    
    #print('true_batch_size: {} \t all_batch_size: {}'.format(true_batch, all_batch_size))
    
    while 1:
        all_patch_list = shuffle(all_patch_list)
        det_patch_list = shuffle(det_patch_list)
        
        det_patch_list_cycle = cycle(det_patch_list)
        
        for offset in range(0,len(all_patch_list),all_batch_size):
            
            ## Get file and coords list from each patch list and combine them
            all_samples = all_patch_list[offset:offset+all_batch_size]
            true_sample = []
            count = 0
            for sample in det_patch_list_cycle:
                true_sample.append(sample)
                count += 1
                if count>=true_batch:
                    break
            combined_sample_list = all_samples
            combined_sample_list.extend(true_sample)
            
            combined_sample_list = shuffle(combined_sample_list)
            
            patch = []
            ground_truth = []
            
            for sample in combined_sample_list:
                filename = folder + sample[0]
                coords = sample[1]
                level = levels[np.random.randint(0, len(levels), dtype=np.int8)]
                patch.append(getRegionFromSlide(getWSI(filename), level=level, start_coord=coords, dims=dims))
                
                ground_truth.append(getLabel(filename,level,coords,dims))
                
                #print('Level used: {}'.format(level))
                
            X_train = np.array(patch)
            y_train = np.array(ground_truth)
            
            yield shuffle(X_train, y_train)