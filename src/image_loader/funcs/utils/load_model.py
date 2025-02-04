import tensorflow.keras.models as tf

def load_model(model_path: str, model_name: str):
    model = tf.load_model(f'models/{model_path}/{model_name}.h5')
    model.summary()
    return model