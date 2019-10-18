import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


def plot_images(images, labels_true, predictions=None):
    # Create a figure with sub-plots
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    # Adjust the vertical spacing
    if predictions is None:
        hspace = 0.2
    else:
        hspace = 0.5
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Fix crash when less than 9 images
        if i < len(images):
            # Plot the image
            ax.imshow(images[i], interpolation='spline16')

            # Name of the true class
            labels_true_name = CLASS_NAMES[labels_true[i][0]]

            # Show true and predicted classes
            if predictions is None:
                xlabel = "True: " + labels_true_name
            else:
                # Name of the predicted class
                labels_pred_name = CLASS_NAMES[predictions[i]]

                xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name

            # Show the class on the x-axis
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plot
    plt.show()


def format_predictions(predictions):
    return np.argmax(predictions, axis=1)
