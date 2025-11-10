
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input as resnet
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetV2_preprocessing

#from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_eff


class model:
    def init(self, path):
        self.model_MobileNetV2_model = tf.keras.models.load_model(os.path.join(path, 'MobileNetV2_model.hdf5'))
        self.model_MobileNetV2_model_1 = tf.keras.models.load_model(os.path.join(path, 'MobileNetV2_model_1.hdf5'))
        self.model_EfficientnetB0 = tf.keras.models.load_model(os.path.join(path, 'EfficientnetB0.hdf5'))
        self.model_EfficientnetB0_1 = tf.keras.models.load_model(os.path.join(path, 'EfficientnetB0_1.hdf5'))
        self.model_ResNet50 = tf.keras.models.load_model(os.path.join(path, 'ResNet50.hdf5'))

        
    def predict(self, X):
        # Insert your preprocessing here
        X_mobile = mobilenetV2_preprocessing(X)
        X_resnet = resnet(X)
        X_efficient = efficientnet_preprocessing(X)
        yhats = []
        yhats.append(self.model_MobileNetV2_model.predict(X_mobile))
        yhats.append(self.model_MobileNetV2_model_1.predict(X_mobile))
        yhats.append(self.model_EfficientnetB0.predict(X_efficient))
        yhats.append(self.model_EfficientnetB0_1.predict(X_efficient))
        yhats.append(self.model_ResNet50.predict(X_resnet))
        yhats = np.array(yhats)
        # sum across ensemble members
        summed = np.sum(yhats, axis=0)
        # argmax across classes
        out = tf.argmax(summed, axis=-1)
        return out