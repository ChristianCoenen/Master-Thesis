from epn.network_supervised import EPNetworkSupervised

# Configure and train the Entropy Propagation Network
epn = EPNetworkSupervised(
    dataset="mnist",
    encoder_dims=[1024, 512, 256],
    discriminator_dims=[1024, 512, 256],
    latent_dim=50,
    classification_dim=10,
    weight_sharing=True,
    autoencoder_loss=["mean_squared_error", "binary_crossentropy"],
)
# Only run the following line if you have graphviz installed, otherwise make sure to remove it or comment it out
epn.save_model_architecture_images()

epn.train_autoencoder(epochs=3, batch_size=32, validation_split=0.1)
epn.train(epochs=40, batch_size=128, steps_per_epoch=500, train_encoder=True)
