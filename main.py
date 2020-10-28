from epn import datasets
from epn.network_supervised import EPNetworkSupervised
import tensorflow as tf

tf.random.set_seed(30)
data = datasets.get_mnist(fashion=False)
# Configure and train the Entropy Propagation Network
epn = EPNetworkSupervised(
    data=data,
    latent_dim=50,
    autoencoder_loss=["categorical_crossentropy", "binary_crossentropy"],
    weight_sharing=True,
    encoder_dims=[500, 500],
    discriminator_dims=[500, 500],
)
# Only run the following line if you have graphviz installed, otherwise make sure to remove it or comment it out
epn.save_model_architecture_images()

epn.visualize_autoencoder_predictions_to_file(state="pre_autoencoder_training")
epn.train_autoencoder(epochs=10, batch_size=32, validation_split=0.1)
acc = epn.evaluate()
epn.visualize_autoencoder_predictions_to_file(state="post_autoencoder_training", acc=acc)

epn.train(epochs=40, batch_size=128, steps_per_epoch=500, train_encoder=True)
