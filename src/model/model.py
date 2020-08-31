# import keras
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.optimizers import Adam

from .evaluation import calculate_IOU, calculate_loss

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
            })
    
    def save(self, filepath):
        self.model.save(filepath)
    
    
    def build(self, input_shape, output_shape, hyperparams):
        alpha = 0.2
        dropout = 0.1

        layers = [
            Conv2D(16, kernel_size=(3, 3), strides=1, input_shape=input_shape),
            LeakyReLU(alpha=alpha),
            # Dropout(dropout),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(32, kernel_size=(3, 3), strides=1),
            LeakyReLU(alpha=alpha),
            # Dropout(dropout),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(32, kernel_size=(3, 3), strides=1),
            LeakyReLU(alpha=alpha),
            # Dropout(dropout),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, kernel_size=(3, 3), strides=1),
            LeakyReLU(alpha=alpha),
            # Dropout(dropout),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, kernel_size=(3, 3), strides=1),
            LeakyReLU(alpha=alpha),
            # Dropout(dropout),
            Conv2D(128, kernel_size=(3, 3), strides=1),
            LeakyReLU(alpha=alpha),
            # Dropout(dropout),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(120),
            LeakyReLU(alpha=alpha),
            # Dropout(dropout),
            Dense(62),
            LeakyReLU(alpha=alpha),
            # Dropout(dropout),
            Dense(output_shape[0]),
            LeakyReLU(alpha=alpha),
        ]

        model = Sequential(layers)

        model.compile(
            optimizer=Adam(lr=hyperparams.get("learning_rate", 0.0001)),
            loss=calculate_loss,
            metrics=hyperparams.get("metrics", [calculate_IOU, "accuracy"])
        )
        return model
