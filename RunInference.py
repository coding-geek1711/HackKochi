from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
# import pickle

# print('Thanks to almighty! \ntf {}'.format(tf._version_))


def demo_results(model, image_path, class_names, target_size=(128, 128, 3)):
    '''
    Demo: (load image -> predict class -> show result)

    Required: model, img_path, class_names, target_size (default: (224, 224, 3)) 
    '''
    img = image.load_img(image_path, color_mode='rgb', target_size=target_size)
    img = image.img_to_array(img).astype('float32') / 255.0
    #
    class_ = np.argmax(model.predict(np.expand_dims(img, axis=0))[0], axis=-1)
    predicted_label = class_names[class_]
    #
    print(predicted_label)

    return predicted_label
