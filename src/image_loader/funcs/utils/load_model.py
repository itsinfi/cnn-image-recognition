import tensorflow.keras.models as tf

def load_model(model_name: str):
    model = tf.load_model(f'{model_name}.h5')
    model.summary()
    return model