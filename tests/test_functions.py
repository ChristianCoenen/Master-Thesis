from numpy.testing import assert_array_equal
import numpy as np


def get_weights_no_bias(layers):
    # Get all weight matrices from a specific layer, ignore the bias arrays
    weights = {}
    for layer in layers:
        for i, weight_var in enumerate(layer.weights):
            if "kernel" in weight_var.name:
                weights[weight_var.name] = layer.get_weights()[i]
    return weights


def verify_shared_weights(autoencoder, encoder, decoder):
    # Get all encoder & decoder weights (no biases, cause they can't be shared anyway)
    encoder_weights = sorted(get_weights_no_bias(encoder.layers))
    decoder_weights = sorted(get_weights_no_bias(decoder.layers))
    autoencoder_weights = sorted(get_weights_no_bias(autoencoder.layers))

    # Verify that all three models have the same number of weights matrices (excluding bias) to verify weight sharing
    assert len(encoder_weights) == len(decoder_weights) == len(autoencoder_weights)

    # Verify that the autoencoder & decoder are having the same weights as the encoder
    # Note that the Autoencoder consists of the encoder and decoder model
    for i in range(len(encoder_weights)):
        assert_array_equal(decoder_weights[i], encoder_weights[i], autoencoder_weights[i])


def get_num_trainable_parameters(autoencoder):
    return np.sum([np.prod(v.shape) for v in autoencoder.trainable_variables])
