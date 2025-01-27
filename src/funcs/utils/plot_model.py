from tensorflow.keras.utils import plot_model as plot
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt 

file_name = 'model_structure.png'

def plot_model(model: Sequential):
    plot(model, to_file=file_name, show_shapes=True, show_layer_names=True)

    img = plt.imread(file_name)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()