import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers import LeakyReLU

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

        inputs = Input(shape=input_shape)

        x = Conv2D(16, (3, 3))(inputs)
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = SeparableConv2D(32, (3, 3), padding="same")(x)
        # x = keras.layers.add([y, x])
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        # y = LeakyReLU(alpha)(y)
        # x = keras.layers.add([y, x])
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = SeparableConv2D(64, (3, 3), padding="same")(x)
        # x = keras.layers.add([y, x])
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        # x = keras.layers.add([y, x])
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = SeparableConv2D(128, (3, 3), padding="same")(x)
        # x = keras.layers.add([y, x])
        # x = BatchNormalization()(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)

        x = Dense(64)(x)
        x = LeakyReLU(alpha)(x)
        x = Dense(output_shape[0])(x)
        outputs = LeakyReLU(alpha)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(lr=hyperparams.get("learning_rate", 0.0001)),
            loss=calculate_loss,
            metrics=hyperparams.get("metrics", [calculate_IOU, "accuracy"])
        )
        return model
