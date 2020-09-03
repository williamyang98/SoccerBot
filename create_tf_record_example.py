import tensorflow as tf

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, image_name, x_center, y_center, confidence):
    feature = {
      'image': _bytes_feature(image),
      'image_name': _bytes_feature(image_name),
      'x_center': _float_feature(x_center),
      'y_center': _float_feature(y_center),
      'confidence': _float_feature(confidence)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()