import pickle

if __name__ == "__main__":
    #TODO: a simple example to load a saliency dataset file
    #RETURN: the values of the first record of the file. The record includes: timestamp, fixation_list, and saliency_map
    
    #load the dataset file named `saliency_ds1_topicparis` (ds=1, video=paris)
    data = pickle.load(open('./data/saliency_ds1_topicparis'))
    
    #access the first record
    timestamp, fixation_list, saliency_map = data[0]
    
    #print out the values of fields in the first record
    print timestamp
    print fixation_list
    print saliency_map
    
    
