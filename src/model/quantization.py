import os
import tensorflow as tf

from model import Model
from paths import ASSETS_DIR, MODEL_DIR

def main():
    model_weights_filepath = os.path.join(MODEL_DIR, "model-weights.h5")
    output_filepath = os.path.join(MODEL_DIR, "quantized-model.tflite")

    # create using weights
    model = Model((256,256,3), (4,))
    model.load_weights(model_weights_filepath)

    # quantize and store as bytefile
    quantized_model = convert(model.model)
    with open(output_filepath, 'wb+') as file:
        file.write(quantized_model)

def convert(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # converter.target_spec.supported_types = [tf.lite.constants.FLOAT16] # reduce weights
    tflite_quant_model = converter.convert()
    return tflite_quant_model

if __name__ == '__main__':
    main()



