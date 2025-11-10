import matplotlib.pyplot as plt
from typing import Optional

def show_images(dataset, labels, batch_no, no_images_per_batch, num_cols:int =10):

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


def plot_history(history, title: Optional[str] = None):
    """
    Plot training & validation accuracy/loss from a Keras History object.
    Compatible with: plot_history(history, title="...").
    """
    if history is None or getattr(history, "history", None) is None:
        print("plot_history: no history to plot.")
        return

    hist = history.history
    acc = hist.get("accuracy")
    val_acc = hist.get("val_accuracy")
    loss = hist.get("loss")
    val_loss = hist.get("val_loss")

    if not acc and not loss:
        print("plot_history: missing 'accuracy'/'loss' keys.")
        return

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    if acc:
        plt.plot(range(1, len(acc) + 1), acc, label="Train")
    if val_acc:
        plt.plot(range(1, len(val_acc) + 1), val_acc, label="Val")
    plt.title(f"Accuracy{(' — ' + title) if title else ''}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if acc or val_acc:
        plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    if loss:
        plt.plot(range(1, len(loss) + 1), loss, label="Train")
    if val_loss:
        plt.plot(range(1, len(val_loss) + 1), val_loss, label="Val")
    plt.title(f"Loss{(' — ' + title) if title else ''}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if loss or val_loss:
        plt.legend()

    plt.tight_layout()
    plt.show()