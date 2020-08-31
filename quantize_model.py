import os
import argparse

MODEL_DIR = "assets/models/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", default=os.path.join(MODEL_DIR, "cnn_113_80.h5f"))
    parser.add_argument("--model-out", default=os.path.join(MODEL_DIR, "quantized-model.tflite"))
    parser.add_argument("--large", action="store_true")

    args = parser.parse_args()

    from load_model import load_from_filepath
    model, (HEIGHT, WIDTH) = load_from_filepath(args.model_in, args.large)


    # quantize and store as bytefile
    quantized_model = convert(model)
    with open(args.model_out, 'wb+') as file:
        file.write(quantized_model)

def convert(model):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # converter.target_spec.supported_types = [tf.lite.constants.FLOAT16] # reduce weights
    tflite_quant_model = converter.convert()
    return tflite_quant_model

if __name__ == '__main__':
    main()



