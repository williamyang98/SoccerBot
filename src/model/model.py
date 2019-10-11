import tensorflow as tf
import tensorflow.keras as keras

from evaluation import calculate_IOU, calculate_loss

class Model:
    def __init__(self, input_shape=(256, 256, 3), output_shape=(1,4), hyperparams={}):
        self.model = self.build_model(input_shape, output_shape, hyperparams)
    
    def fit(self, X, Y, hyperparams):
        self.model.fit(X, Y,
            batch_size=hyperparams.get('batch_size', 100),
            epochs=hyperparams.get('epochs', 10),
            validation_data=hyperparams.get('validation_data'))
        
        self.model.summary()
    
    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath):
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model = keras.models.load_model(filepath)
    
    def build_model(self, input_shape, output_shape, hyperparams):
        alpha = 0.2

        layers = [
			keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, input_shape=input_shape),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

			keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # flatten for fully connected network
			keras.layers.Flatten(),

			keras.layers.Dense(120),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(40),
			keras.layers.LeakyReLU(alpha=alpha),
			keras.layers.Dense(4),
			keras.layers.LeakyReLU(alpha=alpha),
        ]

        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.Adam(lr=hyperparams.get("learning_rate", 0.0001)),
            loss=calculate_loss,
            metrics=[calculate_IOU]
        )
        return model