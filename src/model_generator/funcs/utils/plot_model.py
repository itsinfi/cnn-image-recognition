from tensorflow.keras.utils import plot_model as plot
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt 


def plot_model(model: Sequential, model_name: str):
    file_name = f'models/{model_name}/{model_name}_structure.png'

    plot(model, to_file=file_name, show_shapes=True, show_layer_names=True)

    img = plt.imread(file_name)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()