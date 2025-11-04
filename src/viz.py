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
