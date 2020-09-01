import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="assets/models/cnn_227_160_quantized.tflite")
    parser.add_argument("--checkpoint", action="store_true")
    # parser.add_argument("--model", default="assets/models/cnn_113_80.h5f")
    # parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--large", action="store_true")

    args = parser.parse_args()

    from load_model import load_any_model_filepath
    model, (HEIGHT, WIDTH) = load_any_model_filepath(args.model, not args.checkpoint, args.large)

    import tensorflow as tf
    from load_tf_records import get_test_dataset

    record_filenames = sorted(tf.io.gfile.glob("./assets/data/records/images-*.tfrec"))

    dataset = get_test_dataset(record_filenames, (HEIGHT, WIDTH))

    BATCH_SIZE = 128
    dataset = dataset.batch(BATCH_SIZE)

    def detect_accuracy(y_true, y_pred, thresh=0.8):
        true_cls = y_true[:,2]
        pred_cls = y_pred[:,2]
        pred_cls = tf.math.greater(pred_cls, thresh)
        pred_cls = tf.cast(pred_cls, true_cls.dtype)
        
        abs_err = tf.math.abs(true_cls-pred_cls)
        return 1-tf.math.reduce_mean(abs_err)

    def position_accuracy(y_true, y_pred):
        true_cls = y_true[:,2]
        true_pos = y_true[:,:2]
        pred_pos = y_pred[:,:2]
        
        abs_err = tf.math.abs(true_pos-pred_pos)
        dist_sqr_err = tf.math.reduce_sum(tf.math.square(abs_err), axis=1)
        dist_err = tf.math.sqrt(dist_sqr_err)
        
        # only consider when object is there
        dist_err = tf.math.multiply(dist_err, true_cls)
        net_err = tf.math.reduce_sum(dist_err)
        total_objects = tf.math.reduce_sum(true_cls)
        
        mean_err = net_err / tf.math.maximum(total_objects, 1)
        return 1-mean_err     

    metrics = []

    record_size = 10000
    N = len(record_filenames)*record_size
    total_batches = N // BATCH_SIZE
    
    for batch_id, (images, labels) in enumerate(dataset.take(total_batches)):
        batch_metrics = []
        for i in range(BATCH_SIZE):
            image = images[i:i+1]
            y_true = labels[i:i+1]
        
            y_pred = model.predict(image)
            detect_acc = detect_accuracy(y_true, y_pred)
            pos_acc = position_accuracy(y_true, y_pred)


            batch_metrics.append((detect_acc, pos_acc))
            metrics.append((detect_acc, pos_acc))

        detect_acc = sum([m[0] for m in batch_metrics]) / len(batch_metrics)
        pos_acc = sum([m[1] for m in batch_metrics]) / len(batch_metrics)
        print(f"batch: {batch_id:3d}/{total_batches}, detect_accuracy: {detect_acc:.3f}, position_accuracy: {pos_acc:.3f}\r", end='')
    
    detect_acc = sum([m[0] for m in metrics]) / len(metrics)
    pos_acc = sum([m[1] for m in metrics]) / len(metrics)

    print()
    print(">> Results summary")
    print(f"detect_accuracy: {detect_acc:.2f} position_accuracy: {pos_acc:.2f}")

if __name__ == '__main__':
    main()


    

    






