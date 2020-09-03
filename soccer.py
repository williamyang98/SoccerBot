import argparse
from src.app import App, Predictor, MSSScreenshot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="assets/models/cnn_227_160_quantized.tflite")
    parser.add_argument("--checkpoint", action="store_true")
    # parser.add_argument("--model", default="assets/models/cnn_113_80.h5f")
    # parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--large", action="store_true")

    parser.add_argument("--preview", action='store_true')
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    from load_model import load_any_model_filepath
    model, (HEIGHT, WIDTH) = load_any_model_filepath(args.model, not args.checkpoint, args.large)

    # screen box is (x, y, width, height)
    app = App(args.debug, args.preview)
    # app.bounding_box = (677, 289, 325, 500)
    # app.bounding_box = (677, 289, 400, 550)
    app.bounding_box = (677, 289, 322, 455)

    screenshotter = MSSScreenshot()
    # screenshotter = D3DScreenshot()

    predictor = Predictor(model, (HEIGHT, WIDTH))
    predictor.acceleration = 5

    app.start(predictor, screenshotter)


if __name__ == "__main__":
    main()
