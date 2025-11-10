import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetV2_preprocessing


class model:
    def init(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range = 30,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'reflect',
            preprocessing_function=mobilenetV2_preprocessing
        )

    def predict(self, X, num_augmentations=5):
        augmented_predictions = []

        for _ in range(num_augmentations):
            # Apply augmentation to the input data
            augmented_X = np.array([self.datagen.random_transform(img) for img in X])
            augmented_X = self.datagen.standardize(augmented_X)

            # Make prediction on the augmented data
            out = self.model.predict(augmented_X)
            augmented_predictions.append(out)

        # Use majority voting to determine the final prediction
        yhats = np.array(augmented_predictions)
        # sum across ensemble members
        summed = np.sum(yhats, axis=0)
        # argmax across classes
        out = tf.argmax(summed, axis=-1)

        return out