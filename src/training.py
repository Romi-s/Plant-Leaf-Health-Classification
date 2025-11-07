

from .models.cnn import build_cnn_tuned, get_cnn_callbacks
from .models.mobilenetv2 import build_mobilenetv2_model, get_mobilenetv2_callbacks
from .models.resnet50 import build_resnet50_model, get_resnet50_callbacks
from .models.efficientnetb0 import build_efficientnetb0_model, get_efficientnetb0_callbacks
from .models.inceptionv3 import build_inceptionv3_model, get_inceptionv3_callbacks
# Training functions for new models
def train_resnet50(train_gen, val_gen, epochs=50, hidden_units=128, verbose=1):
	model = build_resnet50_model(hidden_units=hidden_units)
	callbacks = get_resnet50_callbacks()
	return train_model(model, train_gen, val_gen, epochs, callbacks, verbose)

def train_efficientnetb0(train_gen, val_gen, epochs=60, hidden_units=512, hidden_units_1=256, verbose=1):
	model = build_efficientnetb0_model(hidden_units=hidden_units, hidden_units_1=hidden_units_1)
	callbacks = get_efficientnetb0_callbacks()
	return train_model(model, train_gen, val_gen, epochs, callbacks, verbose)

def train_inceptionv3(train_gen, val_gen, epochs=60, verbose=1):
	model = build_inceptionv3_model()
	callbacks = get_inceptionv3_callbacks()
	return train_model(model, train_gen, val_gen, epochs, callbacks, verbose)
from .data import load_npz_dataset, encode_labels, split_dataset
from .utils.viz import plot_training_history


def train_model(model, train_gen, val_gen, epochs=50, callbacks=None, verbose=1):
	"""
	Train any Keras model with provided generators and callbacks.
	Args:
		model: compiled Keras model
		train_gen: training data generator
		val_gen: validation data generator
		epochs: number of epochs
		callbacks: list of callbacks
		verbose: verbosity level
	Returns:
		Training history
	"""
	history = model.fit(
		train_gen,
		epochs=epochs,
		verbose=verbose,
		callbacks=callbacks,
		validation_data=val_gen
	)
	plot_training_history(history)
	return history


def train_cnn(train_gen, val_gen, epochs=50, hp=None, verbose=1):
	model = build_cnn_tuned(hp) if hp is not None else build_cnn_tuned(None)
	callbacks = get_cnn_callbacks()
	return train_model(model, train_gen, val_gen, epochs, callbacks, verbose)


def train_mobilenetv2(train_gen, val_gen, epochs=50, verbose=1):
	model = build_mobilenetv2_model()
	callbacks = get_mobilenetv2_callbacks()
	return train_model(model, train_gen, val_gen, epochs, callbacks, verbose)


# Example main workflow
if __name__ == "__main__":
	# Load data
	images, labels = load_npz_dataset("data/public_data.npz")
	binary_labels, label_encoder = encode_labels(labels)
	aug_train_set, validation_set, class_weights = split_dataset(images, binary_labels)

	# Train CNN
	print("Training CNN...")
	history_cnn = train_cnn(aug_train_set, validation_set, epochs=50)

	# Train MobileNetV2
	print("Training MobileNetV2...")
	history_mobilenet = train_mobilenetv2(aug_train_set, validation_set, epochs=50)

	# Train ResNet50
	print("Training ResNet50...")
	history_resnet50 = train_resnet50(aug_train_set, validation_set, epochs=50)

	# Train EfficientNetB0
	print("Training EfficientNetB0...")
	history_efficientnetb0 = train_efficientnetb0(aug_train_set, validation_set, epochs=60)

	# Train InceptionV3
	print("Training InceptionV3...")
	history_inceptionv3 = train_inceptionv3(aug_train_set, validation_set, epochs=60)
