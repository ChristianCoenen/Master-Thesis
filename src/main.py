import tensorflow as tf
from src import test_functions
from src.epn import EntropyPropagationNetwork

# Config GPU if available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(physical_devices)}")
tf.config.experimental.set_memory_growth(physical_devices[0], True) if physical_devices else None

# Configure and train the Entropy Propagation Network
epn = EntropyPropagationNetwork()
epn.train()
epn.show_reconstructions(epn.x_train_norm)
epn.show_fake_samples(n_samples=10)

''' Tests '''
# verify generator weight sharing after training
if epn.weight_sharing:
    test_functions.verify_shared_weights(autoencoder=epn.autoencoder, encoder=epn.encoder, decoder=epn.decoder)
