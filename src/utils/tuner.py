from keras_tuner import RandomSearch
# Example usage for CNN model tuning
# You can import your model builder from models/cnn.py
# from src.models.cnn import build_cnn_tuned


def run_cnn_tuner(model_builder, train_gen, val_gen, epochs=50, max_trials=10, project_name='cnn_tuning', directory='tuner_results'):
    """
    Run Keras Tuner RandomSearch for a CNN model.
    Args:
        model_builder: function that builds a Keras model (with hp argument)
        train_gen: training data generator
        val_gen: validation data generator
        epochs: number of epochs per trial
        max_trials: number of hyperparameter search trials
        project_name: tuner project name
        directory: tuner results directory
    Returns:
        Best Keras model found
    """
    tuner = RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=max_trials,
        directory=directory,
        project_name=project_name
    )
    tuner.search(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model, tuner
