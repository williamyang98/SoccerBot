import tensorflow as tf
import tensorflow.keras as keras

class Model:
    def __init__(self, input_shape, output_shape, hyperparams={}):
        self.model = self.build_model(input_shape, output_shape, hyperparams)
    
    def fit(self, X, Y, hyperparams):
        self.model.fit(X, Y,
            batch_size=hyperparams.get('batch_size', 100),
            epochs=hyperparams.get('epochs', 10),
            validation_data=hyperparams.get('validation_data'))
        
        self.summary()

    def summary(self):
        self.model.summary()
    
    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath):
        self.model.save_weights(filepath)
    
    def load(self, filepath):
        self.model.load_weights(filepath)
    
    def build_model(self, input_shape, output_shape, hyperparams):
        alpha = 0.2

        layers = [
            keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu", strides=1, input_shape=input_shape),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", strides=1),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", strides=1),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", strides=1),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", strides=1),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),

            keras.layers.Dense(1240, activation="relu"),
            keras.layers.Dense(640, activation="relu"),
            keras.layers.Dense(480, activation="relu"),
            keras.layers.Dense(120, activation="relu"),
            keras.layers.Dense(62, activation="relu"),
            keras.layers.Dense(4, activation="relu"),
        ]

        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.Adam(lr=hyperparams.get("learning_rate", 0.0001)),
            loss=hyperparams.get("loss", "mse"),
            metrics=hyperparams.get("metrics", ["accuracy"])
        )
        return model
