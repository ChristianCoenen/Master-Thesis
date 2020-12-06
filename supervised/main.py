import sys
import os
sys.path.insert(0, os.getcwd())
from network import datasets
from supervised.network_supervised import NetworkSupervised
import network.helper as helper

seed_value = 30
helper.set_seeds(seed_value)


data = datasets.get_mnist(fashion=False)
# Configure and train the Entropy Propagation Network
epn = NetworkSupervised(
    data=data,
    latent_dim=20,
    autoencoder_loss=["categorical_crossentropy", "binary_crossentropy"],
    weight_sharing=True,
    encoder_dims=[1024, 500, 256],
    discriminator_dims=[1024, 500, 256],
    seed=seed_value,
)
# Only run the following line if you have graphviz installed, otherwise make sure to remove it or comment it out
epn.save_model_architecture_images()

epn.train(epochs=40, batch_size=128, steps_per_epoch=500, train_encoder=True)
acc = epn.evaluate()
epn.visualize_autoencoder_predictions_to_file(state="post_gan_training", acc=acc)
epn.create_modified_classification_plot(sample_idx=4)
epn.create_modified_classification_plot(random=True)
