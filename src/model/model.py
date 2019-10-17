import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D

from .evaluation import calculate_IOU, calculate_loss, calculate_confidence_error

class Model:
    def __init__(self, input_shape, output_shape, hyperparams={}):
        self.model = self.build(input_shape, output_shape, hyperparams)
    
    def fit(self, X, Y, hyperparams):
        self.model.fit(X, Y,
            batch_size=hyperparams.get('batch_size', 100),
            epochs=hyperparams.get('epochs', 10),
            validation_data=hyperparams.get('validation_data'))

    def fit_generator(self, *args, **kwargs):
        self.model.fit_generator(*args, **kwargs)

    def summary(self):
        self.model.summary()
    
    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def load(self, filepath):
        self.model = keras.models.load_model(
            filepath, 
            custom_objects={
                "calculate_IOU": calculate_IOU,
                "calculate_loss": calculate_loss,
                "calculate_confidence_error": calculate_confidence_error
            })
    
    def save(self, filepath):
        self.model.save(filepath)
    
    
    def build(self, input_shape, output_shape, hyperparams):
        alpha = 0.2
        dropout = 0.1

        layers = [
            keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, input_shape=input_shape),
            keras.layers.LeakyReLU(alpha=alpha),
            keras.layers.Dropout(dropout),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
            keras.layers.LeakyReLU(alpha=alpha),
            keras.layers.Dropout(dropout),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
            keras.layers.LeakyReLU(alpha=alpha),
            keras.layers.Dropout(dropout),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            #keras.layers.Conv2D(128, kernel_size=(3, 3),  strides=1),
            #keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # keras.layers.Conv2D(256, kernel_size=(3, 3),  strides=1),
            # keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),

            # keras.layers.Dense(1240g),
            # keras.layers.Dense(640g),
            #keras.layers.Dense(480g),
            keras.layers.Dense(120),
            keras.layers.LeakyReLU(alpha=alpha),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(62),
            keras.layers.LeakyReLU(alpha=alpha),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(output_shape[0]),
            keras.layers.LeakyReLU(alpha=alpha),
        ]

        model = keras.Sequential(layers)

        model.compile(
            optimizer=keras.optimizers.Adam(lr=hyperparams.get("learning_rate", 0.0001)),
            loss=calculate_loss,
            metrics=hyperparams.get("metrics", [calculate_IOU, calculate_confidence_error, "accuracy"])
        )
        return model
