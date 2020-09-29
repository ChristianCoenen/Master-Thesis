from epn.network import EntropyPropagationNetwork

# Configure and train the Entropy Propagation Network
epn = EntropyPropagationNetwork(
    dataset="mnist",
    encoder_dims=[1024, 512, 256],
    latent_dim=50,
    classification_dim=10,
    weight_sharing=True,
    autoencoder_loss=["mean_squared_error", "binary_crossentropy"],
    graphviz_installed=True,
)
epn.train(epochs=40, batch_size=128, pre_train_epochs=3, train_encoder=True)
