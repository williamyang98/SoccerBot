import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

class LiteModel:
    def __init__(self, model_content):
        model_content = bytes(model_content)
        self.interpreter = Interpreter(model_content=model_content)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]['index']
        self.output_index = output_details[0]['index']

        self.input_scale, self.input_zero_point = input_details[0]['quantization']
        self.output_scale, self.output_zero_point = output_details[0]['quantization']

        self.interpreter.allocate_tensors()
    
    def predict(self, X):
        X = self.input_map(X)
        self.interpreter.set_tensor(self.input_index, X)
        self.interpreter.invoke()
        Y = self.interpreter.get_tensor(self.output_index)
        Y = self.output_map(Y)
        return Y
    
    def input_map(self, x):
        return np.array(x, dtype=np.float32)
        # return np.array(x / self.input_scale + self.input_zero_point, dtype=np.uint8)
    
    def output_map(self, y):
        return np.array(y, dtype=np.float32)
        # y = np.array(y, dtype=np.float32)
        # return (y - self.output_zero_point) * self.output_scale