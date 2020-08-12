from src.epn import EntropyPropagationNetwork
from tests import test_functions
import pytest
import numpy as np


def test_shared_weights():
    epn = EntropyPropagationNetwork(dataset='mnist', weight_sharing=True,
                                    encoder_dims=[100], latent_dim=40, classification_dim=10)
    epn.train(epochs=1)
    test_functions.verify_shared_weights(autoencoder=epn.autoencoder, encoder=epn.encoder, decoder=epn.decoder)


def test_num_trainable_weights_with_sharing():
    encoder_dims, latent_dim, classification_dim = ([400, 200, 100], 40, 10)
    epn = EntropyPropagationNetwork(dataset='mnist', weight_sharing=True, encoder_dims=encoder_dims,
                                    latent_dim=latent_dim, classification_dim=classification_dim)
    res = (np.prod(epn.encoder.input_shape[1:3]) * encoder_dims[0]) + 2 * encoder_dims[0]
    for i in range(len(encoder_dims) - 1):
        res += (encoder_dims[i] * encoder_dims[i+1]) + 2 * encoder_dims[i+1]

    res += (encoder_dims[-1] * (latent_dim + classification_dim)) + latent_dim + classification_dim
    # Bias to last layer
    res += np.prod(epn.encoder.input_shape[1:3])
    assert res == test_functions.get_num_trainable_parameters(epn.autoencoder)


def test_num_trainable_weights_without_sharing():
    encoder_dims, latent_dim, classification_dim = ([400, 200, 100], 40, 10)
    epn = EntropyPropagationNetwork(dataset='mnist', weight_sharing=False, encoder_dims=encoder_dims,
                                    latent_dim=latent_dim, classification_dim=classification_dim)
    res = 2 * (np.prod(epn.encoder.input_shape[1:3]) * encoder_dims[0]) + 2 * encoder_dims[0]
    for i in range(len(encoder_dims) - 1):
        res += 2 * (encoder_dims[i] * encoder_dims[i+1]) + 2 * encoder_dims[i+1]

    res += 2 * (encoder_dims[-1] * (latent_dim + classification_dim)) + latent_dim + classification_dim
    # Bias to last layer
    res += np.prod(epn.encoder.input_shape[1:3])
    assert res == test_functions.get_num_trainable_parameters(epn.autoencoder)

def test_invalid_dataset_parameter():
    with pytest.raises(ValueError) as err:
        EntropyPropagationNetwork(dataset='whatever')
        assert 'Unknown dataset!' in str(err.value)


def test_epn_with_weight_sharing_mnist():
    c_threshold = 0.90
    d_threshold = 0.75
    epn = EntropyPropagationNetwork(dataset='mnist', weight_sharing=True,
                                    encoder_dims=[100], latent_dim=40, classification_dim=10)
    epn.train(epochs=1)
    _, _, _, c_acc, d_acc = epn.evaluate()
    assert c_acc > c_threshold, f"Classification accuracy: {c_acc}% | Threshold: {c_threshold}%"
    assert c_acc > d_threshold, f"Decoder accuracy: {c_acc}% | Threshold: {d_threshold}%"


def test_epn_without_weight_sharing_mnist():
    c_threshold = 0.90
    d_threshold = 0.75
    epn = EntropyPropagationNetwork(dataset='mnist', weight_sharing=False,
                                    encoder_dims=[100], latent_dim=40, classification_dim=10)
    epn.train(epochs=1)
    _, _, _, c_acc, d_acc = epn.evaluate()
    assert c_acc > c_threshold, f"Classification accuracy: {c_acc}% | Threshold: {c_threshold}%"
    assert c_acc > d_threshold, f"Decoder accuracy: {c_acc}% | Threshold: {d_threshold}%"


def test_epn_with_weight_sharing_fashion_mnist():
    c_threshold = 0.80
    d_threshold = 0.45
    epn = EntropyPropagationNetwork(dataset='fashion_mnist', weight_sharing=True,
                                    encoder_dims=[100], latent_dim=40, classification_dim=10)
    epn.train(epochs=1)
    _, _, _, c_acc, d_acc = epn.evaluate()
    assert c_acc > c_threshold, f"Classification accuracy: {c_acc}% | Threshold: {c_threshold}%"
    assert c_acc > d_threshold, f"Decoder accuracy: {c_acc}% | Threshold: {d_threshold}%"


def test_epn_with_multiple_encoder_decoder_layers():
    c_threshold = 0.85
    d_threshold = 0.75
    epn = EntropyPropagationNetwork(dataset='mnist', weight_sharing=True,
                                    encoder_dims=[400, 200, 100], latent_dim=40, classification_dim=10)
    epn.train(epochs=1)
    _, _, _, c_acc, d_acc = epn.evaluate()
    assert c_acc > c_threshold, f"Classification accuracy: {c_acc}% | Threshold: {c_threshold}%"
    assert c_acc > d_threshold, f"Decoder accuracy: {c_acc}% | Threshold: {d_threshold}%"


def test_epn_with_huge_latent_space():
    c_threshold = 0.85
    d_threshold = 0.75
    epn = EntropyPropagationNetwork(dataset='mnist', weight_sharing=True,
                                    encoder_dims=[100], latent_dim=400, classification_dim=10)
    epn.train(epochs=1)
    _, _, _, c_acc, d_acc = epn.evaluate()
    assert c_acc > c_threshold, f"Classification accuracy: {c_acc}% | Threshold: {c_threshold}%"
    assert c_acc > d_threshold, f"Decoder accuracy: {c_acc}% | Threshold: {d_threshold}%"


def test_epn_with_tiny_latent_space():
    c_threshold = 0.85
    d_threshold = 0.75
    epn = EntropyPropagationNetwork(dataset='mnist', weight_sharing=True,
                                    encoder_dims=[100], latent_dim=3, classification_dim=10)
    epn.train(epochs=1)
    _, _, _, c_acc, d_acc = epn.evaluate()
    assert c_acc > c_threshold, f"Classification accuracy: {c_acc}% | Threshold: {c_threshold}%"
    assert c_acc > d_threshold, f"Decoder accuracy: {c_acc}% | Threshold: {d_threshold}%"