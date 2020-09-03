import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

def decode_image(image_data, target_size):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    if target_size is not None:
        image = tf.image.resize(image, target_size)
    image = tf.reverse(image, axis=[-1])
    #image = tf.reshape(image, [*TARGET_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example, target_size):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), 
        "image_name": tf.io.FixedLenFeature([], tf.string), 
        "x_center": tf.io.FixedLenFeature([], tf.float32),  
        "y_center": tf.io.FixedLenFeature([], tf.float32),  
        "confidence": tf.io.FixedLenFeature([], tf.float32),  
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'], target_size)
    label = (example['x_center'], example['y_center'], example['confidence'])
    label = [tf.cast(x, tf.float32) for x in label]
    return image, label

def read_dataset(filenames, target_size):
    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

    opt = tf.data.Options()
    opt.experimental_deterministic = False
    dataset = dataset.with_options(opt) 

    def wrapped_reader(example):
        return read_labeled_tfrecord(example, target_size)

    dataset = dataset.map(wrapped_reader, num_parallel_calls=AUTO)
    
    return dataset

def get_train_dataset(filenames, target_size):
    d = read_dataset(filenames, target_size)
#     d = d.map(data_augment, num_parallel_calls=AUTO)
    d = d.repeat()
    d = d.shuffle(2048)
    # d = d.batch(BATCH_SIZE)
    d = d.prefetch(AUTO)
    return d

def get_test_dataset(filenames, target_size):
    d = read_dataset(filenames, target_size)
    # d = d.batch(BATCH_SIZE)
    d = d.cache()
    d = d.prefetch(AUTO)
    return d