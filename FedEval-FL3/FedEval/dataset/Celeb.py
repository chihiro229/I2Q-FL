from .FedDataBase import FedData, shuffle
import numpy as np
import os
import cv2
import tensorflow as tf
from tqdm import tqdm

class cel(FedData):
    '''
    Using train data set images
    '''
    def load_data(self):

        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cel')
        X = []
        y = []

        train_path = os.path.join(data_path, "Train")
        for class_folder in os.listdir(train_path) :
            
            class_path = os.path.join(train_path, class_folder) # num class
            
            for img_filename in tqdm(os.listdir(class_path)) :
                img_path = os.path.join(class_path, img_filename) # image path 
                img = cv2.imread(img_path)

                #convert to (32, 32, 3)
                img = cv2.resize(img, (32, 32))
                img = np.array(img)

                X.append(img)
                y.append(int(class_folder))

        self.num_class = np.max(y) + 1
        y = tf.keras.utils.to_categorical(y, self.num_class)

        X, y = shuffle(X, y) # shuffling after to categorical since shuffle returns the data in float64 format

        return X, y
