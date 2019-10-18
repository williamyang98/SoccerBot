import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers import LeakyReLU

from .evaluation import calculate_IOU, calculate_loss

class Model:
    @staticmethod
    def load(filepath):
        model = keras.models.load_model(
            filepath, 
            custom_objects={
                "calculate_IOU": calculate_IOU,
                "calculate_loss": calculate_loss,
            })
        return model
    
    @staticmethod 
    def build(input_shape, output_shape, hyperparams):
        alpha = 0.2
        dropout = 0.1

        inputs = Input(shape=input_shape)

        x = Conv2D(16, (3, 3))(inputs)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = SeparableConv2D(32, (3, 3))(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(32, (3, 3))(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = SeparableConv2D(64, (3, 3))(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, (3, 3))(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = SeparableConv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)

        x = Dense(128)(x)
        x = LeakyReLU(alpha)(x)

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
