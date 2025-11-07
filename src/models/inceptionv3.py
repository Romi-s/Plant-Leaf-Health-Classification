from keras.applications import InceptionV3
from keras import layers, Model
from keras.initializers import HeUniform
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

def build_inceptionv3_model(input_shape=(96, 96, 3), seed=42):
	base_model = InceptionV3(input_shape=input_shape, include_top=False, weights="imagenet", pooling='avg')
	input_layer = layers.Input(shape=input_shape)
	x = base_model(input_layer)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(512, activation=None, kernel_initializer=HeUniform(seed))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.5)(x)
	x = layers.Dense(128, activation=None, kernel_initializer=HeUniform(seed))(x)
	x = layers.Dropout(0.1)(x)
	x = layers.Dense(64, activation=None, kernel_initializer=HeUniform(seed))(x)
	outputs = layers.Dense(2, activation='softmax')(x)
	model = Model(inputs=input_layer, outputs=outputs, name='InceptionV3_model')
	model.get_layer('inception_v3').trainable = True
	for i, layer in enumerate(model.get_layer('inception_v3').layers[:38]):
		layer.trainable = False
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	return model

def get_inceptionv3_callbacks():
	lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.3, min_lr=0.000001)
	checkpoint = ModelCheckpoint('InceptionV3_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
	return [lr_reduction, checkpoint]
