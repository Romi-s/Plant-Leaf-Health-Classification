import tensorflow as tf
from tensorflow import keras as tfk
from keras import layers as tfkl
from keras.applications.mobilenet_v2 import MobileNetV2 as TFMobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as mobilenetV2_preprocessing
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

def build_mobilenetv2_model(input_shape=(96, 96, 3), seed=42):
	base_model = tfk.applications.MobileNetV2(
		input_shape=input_shape,
		include_top=False,
		weights="imagenet",
		pooling='avg',
	)
	input_layer = tfkl.Input(shape=input_shape)
	x = base_model(input_layer)
	x = tfkl.Dropout(0.5)(x)
	x = tfkl.Dense(512, activation=None, kernel_initializer=tf.keras.initializers.HeUniform(seed))(x)
	x = tfkl.Dropout(0.5)(x)
	x = tfkl.Dense(128, activation='leaky_relu', kernel_initializer=tf.keras.initializers.HeUniform(seed))(x)
	outputs = tfkl.Dense(2, activation='softmax', kernel_initializer=tf.keras.initializers.HeUniform(seed))(x)
	model = tf.keras.Model(inputs=input_layer, outputs=outputs, name='MobileNetV2_model')
	# Freeze base model
	model.get_layer('mobilenetv2_1.00_96').trainable = False
	model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy'])
	return model

def get_mobilenetv2_callbacks():
	lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=10, verbose=1, factor=0.2, min_lr=0.000001)
	checkpoint = ModelCheckpoint('MobileNetV2_model.hdf5', save_best_only=True, monitor='accuracy', mode='max')
	return [lr_reduction, checkpoint]

# Example usage:
# model = build_mobilenetv2_model()
# callbacks = get_mobilenetv2_callbacks()
# history = model.fit(aug_train_set, steps_per_epoch=len(aug_train_set), epochs=50, verbose=1, callbacks=callbacks, validation_data=validation_set)
