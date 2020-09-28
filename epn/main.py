from epn.network import EntropyPropagationNetwork


# Configure and train the Entropy Propagation Network
mode = "maze_memories"

# maze_memories
if mode == "maze_memories":
    epn = EntropyPropagationNetwork(
        dataset="maze_memories",
        dataset_path="/Users/christiancoenen/repos/Thesis/Thesis-Reinforcement-Learning/data/9tiles_maze_memories.npy",
        encoder_dims=[40],
        latent_dim=3,
        classification_dim=9,
        weight_sharing=True,
        graphviz_installed=True,
    )
    epn.train_autoencoder(epochs=100, batch_size=2)


# mnist
if mode == "mnist":
    epn = EntropyPropagationNetwork(
        dataset="mnist",
        encoder_dims=[1024, 512, 256],
        latent_dim=50,
        classification_dim=10,
        weight_sharing=True,
        graphviz_installed=True,
    )
    epn.train(epochs=40, batch_size=128, pre_train_epochs=3, train_encoder=True)
