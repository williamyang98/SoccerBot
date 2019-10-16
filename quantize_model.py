import os
import tensorflow as tf
import argparse

from src.model import Model

MODEL_DIR = "assets/model/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", default=os.path.join(MODEL_DIR, "model.h5"))
    parser.add_argument("--model-out", default=os.path.join(MODEL_DIR, "quantized-model.tflite"))

    args = parser.parse_args()

    # create using weights
    model = Model((256,256,3), (5,))
    model.load(args.model_in)

    # quantize and store as bytefile
    quantized_model = convert(model.model)
    with open(args.model_out, 'wb+') as file:
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



