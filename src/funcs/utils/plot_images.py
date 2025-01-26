import numpy as np
import matplotlib.pyplot as plt

def plot_images(images: list[np.ndarray[float]]) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=len(images))
    for j, ax in enumerate(axes):
        ax.matshow(images[j].reshape(28,28), cmap = plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()