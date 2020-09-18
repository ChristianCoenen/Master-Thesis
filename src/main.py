from src.epn import EntropyPropagationNetwork


# Configure and train the Entropy Propagation Network
epn = EntropyPropagationNetwork(encoder_dims=[1024, 512, 256], latent_dim=50, weight_sharing=True)
epn.train(epochs=40, batch_size=128, pre_train_epochs=3, train_encoder=True)
