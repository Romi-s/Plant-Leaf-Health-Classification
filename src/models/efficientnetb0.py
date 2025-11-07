from keras.applications import EfficientNetB0
from keras import layers, Sequential
from keras.initializers import HeUniform
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

def build_efficientnetb0_model(hidden_units=512, hidden_units_1=256, input_shape=(96, 96, 3), seed=42):
	base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
	GAP = layers.GlobalAveragePooling2D()
	dropout = layers.Dropout(0.2, seed=seed)
	dense_layer = layers.Dense(hidden_units, activation=None, kernel_initializer=HeUniform(seed))
	batch_norm = layers.BatchNormalization()
	dropout_1 = layers.Dropout(0.5, seed=seed)
	dense_layer_1 = layers.Dense(hidden_units_1, activation=None, kernel_initializer=HeUniform(seed))
	batch_norm_1 = layers.BatchNormalization()
	dropout_2 = layers.Dropout(0.1, seed=seed)
	prediction_layer = layers.Dense(2, activation='softmax')
	for layer in base_model.layers[:90]:
		layer.trainable = False
	model = Sequential([
		base_model,
		GAP,
		dropout,
		dense_layer,
		batch_norm,
		dropout_1,
		dense_layer_1,
		batch_norm_1,
		dropout_2,
		prediction_layer
	])
	opt = Adam(learning_rate=1e-4)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def get_efficientnetb0_callbacks():
	lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.3, min_lr=0.000001)
	checkpoint = ModelCheckpoint('model_efficientnetB0.hdf5', save_best_only=True, monitor='val_loss', mode='min')
	return [lr_reduction, checkpoint]
