from numpy.testing import assert_array_equal
import numpy as np

def get_weights_no_bias(layer):
    # Get all weight matrices from a specific layer, ignore the bias arrays
    weights = []
    for i, weight_var in enumerate(layer.weights):
        weights.append(layer.get_weights()[i]) if 'kernel' in weight_var.name else None
    return weights


def verify_shared_weights(encoder_layer, decoder_layer, classification_layer=None):
    # Get all encoder & decoder weights (no biases, cause they can't be shared anyway)
    encoder_weights = get_weights_no_bias(encoder_layer)
    decoder_weights = get_weights_no_bias(decoder_layer)
    classification_weights = get_weights_no_bias(classification_layer) if classification_layer else None

    # Verify that the layers only have one weight matrix, otherwise shared weights can't be verified
    if classification_layer:
        assert len(encoder_weights) == 1 and len(classification_weights) == 1 and len(decoder_weights) == 2
    else:
        assert len(encoder_weights) == 1 and len(decoder_weights) == 1

    # Verify that weights are the same
    if classification_layer:
        if decoder_weights[0].shape == encoder_weights[0].shape:
            assert_array_equal(decoder_weights[0], encoder_weights[0])
            assert_array_equal(decoder_weights[1], classification_weights[0])
        else:
            assert_array_equal(decoder_weights[1], encoder_weights[0])
            assert_array_equal(decoder_weights[0], classification_weights[0])
    else:
        assert_array_equal(decoder_weights[0], encoder_weights[0])


def get_num_trainable_parameters(autoencoder):
    return np.sum([np.prod(v.shape) for v in autoencoder.trainable_variables])
