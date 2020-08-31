def create_model(input_shape, output_shape, large=False):
    from tensorflow import keras
    from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, Add
    from tensorflow.keras.layers import LeakyReLU, ReLU

    alpha = 0.2
    dropout = 0.1

    inputs = Input(shape=input_shape)

#    x_skip = inputs
    
    x = Conv2D(16, (3, 3))(inputs)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = SeparableConv2D(32, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = SeparableConv2D(64, (3, 3))(x)
    x = ReLU()(x)

    if large:
        x = MaxPooling2D((2, 2))(x)
    else:
        x = MaxPooling2D((3, 3))(x)
    
#     x_skip = Conv2D(64, (16, 16))(x_skip)
#     x_skip = MaxPooling2D((17,17))(x_skip)
#     x = Add()([x, x_skip])

    # larger network
    if large:
        x = Conv2D(64, (3, 3))(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)

        x = SeparableConv2D(128, (3, 3), padding="same")(x)
        x = ReLU()(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    if large:
        x = Dense(128)(x)
        x = ReLU()(x)
    
    x = Dense(64)(x)
    x = ReLU()(x)

    x = Dense(32)(x)
    x = ReLU()(x)

    x = Dense(output_shape)(x)
    outputs = x
    #outputs = LeakyReLU(alpha)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_from_filepath(filepath, large):
    import re
    p = re.compile(r"cnn_(\d+)_(\d+)\.h5f")
    m = p.findall(filepath)
    
    if len(m) == 0:
        raise IOError(f"error: expected model name cnn_H_W.h5f (got {filepath})")
    
    HEIGHT, WIDTH = m[0]
    HEIGHT, WIDTH = int(HEIGHT), int(WIDTH)

    print(f"loading network {HEIGHT}x{WIDTH} large={large}")

    model = create_model((HEIGHT, WIDTH, 3), 3, large=large)
    model.load_weights(filepath)

    print(model.summary())

    return (model, (HEIGHT, WIDTH))

def load_quantized_model_from_filepath(filepath):
    from src.model import LiteModel
    with open(filepath, "rb") as fp:
        return LiteModel(fp.read())

