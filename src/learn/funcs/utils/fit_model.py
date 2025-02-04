from tensorflow.keras import Sequential
from numpy import ndarray

def fit_model(model: Sequential, 
        x_train: ndarray, 
        y_train: ndarray, 
        train_generator: any, 
        steps_per_epoch: int, 
        epochs: int, 
        verbose: int, 
        callbacks: list
    ):
    
    if train_generator is None:
        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks
        )
    else:
        model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks
        )