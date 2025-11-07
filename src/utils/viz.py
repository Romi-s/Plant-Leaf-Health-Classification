import matplotlib.pyplot as plt

def show_images(dataset, labels, batch_no, no_images_per_batch, num_cols=10):

    start_index = batch_no * no_images_per_batch
    end_index = min((batch_no + 1) * no_images_per_batch, len(dataset))
    num_rows = (end_index - start_index + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for i, ax in enumerate(axes.ravel()):
        if start_index + i < end_index:
            image = dataset[start_index + i]
            image = image / 255.0  # Normalize pixel values to [0, 1]
            ax.imshow(image)
            ax.set_title(f"{labels[start_index + i]}",fontsize=10, y=-0.5)
            ax.axis('off')

    for i in range(end_index - start_index, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.show()

# Example usage:
# show_images(dataset=images,labels =binary_labels, batch_no=0,no_images_per_batch=60)

def plot_training_history(history, title_suffix=""):
    """
    Plots training & validation accuracy and loss from a Keras History object.
    Args:
        history: Keras History object from model.fit()
        title_suffix: Optional string to append to plot titles
    """
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy{title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss{title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
