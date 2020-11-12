from typing import List, Optional
from numpy.random import randint
from epn.network import EPNetwork
from epn.helper import add_subplot, save_plot_as_image
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


class EPNetworkRL(EPNetwork):
    """
    A class implementing the Entropy Propagation Network architecture in an reinforcement learning setting:
    Differences between EPNetworkRL and EPNetworkSupervised:

    General:
    1. The environment on which the data was generated needs to be passed in order to visualize the reconstructions, ...

    Autoencoder:
    1. Encoder has more outputs (q_values, ...)

    GAN:
    1. Additional GAN Network (working with the encoder instead of the decoder)
    """

    def __init__(
        self,
        env,
        data,
        latent_dim,
        autoencoder_loss,
        weight_sharing: bool,
        encoder_dims: List[int],
        discriminator_dims: List[int],
        seed: int,
    ):
        """
        :param env: MazeEnv
            MazeEnv object initialized with the same maze & config that was used to generate the dataset.
        """
        super().__init__(weight_sharing, encoder_dims, discriminator_dims, seed)
        self.nr_valid_tiles = np.count_nonzero(env.maze.to_impassable() == 0)
        self.nr_tiles = env.maze.size[0] * env.maze.size[1]
        self.env = env
        self.classification_dim = self.nr_valid_tiles + self.env.action_space.n + 1
        self.latent_dim = latent_dim
        self.autoencoder_loss = autoencoder_loss
        (self.x_train_norm, self.y_train), (self.x_test_norm, self.y_test) = data

        # Build Autoencoder
        self.encoder, self.decoder, self.autoencoder = self.build_autoencoder(
            encoder_input_tensors=[
                Input(shape=self.env.maze.size[0] + self.env.maze.size[1], name="position"),
                Input(shape=self.nr_valid_tiles, name="state"),
                Input(shape=self.env.action_space.n, name="action"),
            ],
            encoder_output_layers=[
                Dense(self.nr_valid_tiles, activation="softmax", name=f"expected_next_state"),
                Dense(1, activation="tanh", name=f"expected_reward"),
                Dense(self.env.action_space.n, activation="softmax", name=f"reconstructed_action"),
                Dense(self.latent_dim, activation=LeakyReLU(alpha=0.2), name="latent_space"),
            ],
            ae_ignored_output_layer_names=["latent_space"],
        )
        self.autoencoder.compile(loss=self.autoencoder_loss, optimizer="adam", metrics=["accuracy"])

        # Build an encoder and a decoder discriminator
        self.dec_discriminator = self.build_discriminator(
            input_tensors=[
                Input(shape=self.env.maze.size[0] + self.env.maze.size[1], name="reconstructed_position"),
                Input(shape=self.nr_valid_tiles, name="reconstructed_state"),
                Input(shape=self.env.action_space.n, name="recustructed_action"),
            ],
            output_layers=[Dense(1, activation="sigmoid", name="real_or_fake")],
            model_name="dec_discriminator",
        )
        self.dec_discriminator.compile(loss=["binary_crossentropy"], optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])

        self.enc_discriminator = self.build_discriminator(
            input_tensors=[
                Input(shape=self.env.maze.size[0] + self.env.maze.size[1], name="position"),
                Input(shape=self.nr_valid_tiles, name="state"),
                Input(shape=self.nr_valid_tiles, name="expected_next_state"),
                Input(shape=1, name="expected_reward"),
                Input(shape=self.env.action_space.n, name="reconstructed_action"),
            ],
            output_layers=[Dense(1, activation="sigmoid", name="real_or_fake")],
            model_name="enc_discriminator",
        )
        self.enc_discriminator.compile(
            loss=["binary_crossentropy"], optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=["accuracy"]
        )

        # Prevent discriminator weight updates during GAN training
        self.dec_discriminator.trainable = False
        self.enc_discriminator.trainable = False

        # Create the normal decoder GAN network that works against the decoder
        self.dec_gan = self.build_gan(self.decoder, self.dec_discriminator, model_name="dec_gan")
        self.dec_gan.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

        # Create the special encoder GAN network that works against the encoder instead of the decoder
        self.enc_gan = self.build_enc_gan(
            self.encoder,
            self.enc_discriminator,
            ignored_layer_names=["q_values", "latent_space"],
        )
        self.enc_gan.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

    def build_enc_gan(self, encoder, discriminator, ignored_layer_names: Optional[List[str]] = None) -> Model:
        inputs = [Input(shape=tensor.shape[1:], name=tensor.name.split(":")[0]) for tensor in encoder.inputs]
        encoded = encoder(inputs)
        # Only use generated outputs as discriminator inputs that are not specified in 'ignored_layer_names'
        discriminator_inputs = [inputs[0], inputs[1]]
        discriminator_inputs.extend(
            [output for output in encoded if output.name.split("/")[1] not in ignored_layer_names]
            if ignored_layer_names
            else encoded
        )
        discriminated = discriminator(discriminator_inputs)
        return Model(inputs, discriminated, name="enc_gan")

    def train_autoencoder(self, **kwargs):
        episodes = self.get_episodes_from_dataset(n=self.x_train_norm.shape[0], random=False)
        position, env_state, action, reward, next_position, next_env_state, done = episodes

        inputs = [position, env_state, action]
        outputs = [next_env_state, reward, action, inputs]

        self.autoencoder.fit(inputs, outputs, **kwargs)

    def get_episodes_from_dataset(self, n: int, random: bool):
        ix = randint(0, self.x_train_norm.shape[0], n) if random else range(0, n)

        position = np.array([*self.x_train_norm[ix, 0]])
        env_state = np.array([*self.x_train_norm[ix, 1]])
        action = np.array([*self.x_train_norm[ix, 2]])
        reward = np.array([*self.y_train[ix, 0]]).reshape(-1, 1)
        next_position = np.array([*self.y_train[ix, 1]])
        next_env_state = np.array([*self.y_train[ix, 2]])
        done = np.array([*self.y_train[ix, 3]]).reshape(-1, 1)

        return position, env_state, action, reward, next_position, next_env_state, done

    def generate_new_states(self, n: int):
        position_arr = np.empty((n, self.env.maze.size[0] + self.env.maze.size[1]), dtype=int)
        env_state_arr = np.zeros((n, self.nr_valid_tiles), dtype=int)
        action_arr = np.empty((n, self.env.action_space.n), dtype=int)

        counter = 0
        while counter < n:
            # Get new position, state, action
            action = np.random.randint(0, self.env.action_space.n)
            action_one_hot = tf.keras.utils.to_categorical(action, num_classes=self.env.action_space.n)
            self.env.reset(randomize_start=True)
            position = [obj for obj in self.env.maze.objects if obj.name == "agent"][0].positions[0]
            position_1_one_hot = tf.keras.utils.to_categorical(position[0], num_classes=self.env.maze.size[0])
            position_2_one_hot = tf.keras.utils.to_categorical(position[1], num_classes=self.env.maze.size[1])
            position_one_hot = np.concatenate((position_1_one_hot, position_2_one_hot))
            env_state_one_hot = self.env.maze.to_valid_obs()
            random_x = [position_one_hot, env_state_one_hot, action_one_hot]

            is_duplicate = False
            for data_arr in self.x_train_norm:
                if (
                    np.array_equal(data_arr[0], random_x[0])
                    and np.array_equal(data_arr[1], random_x[1])
                    and np.array_equal(data_arr[2], random_x[2])
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                position_arr[counter] = random_x[0]
                env_state_arr[counter] = random_x[1]
                action_arr[counter] = random_x[2]
                counter += 1

        return position_arr, env_state_arr, action_arr

    def train(self, epochs: int, batch_size: int, steps_per_epoch: int, train_encoder: bool):
        half_batch = int(batch_size / 2)
        # manually enumerate epochs
        for i in range(epochs):
            # enumerate batches over the training set
            for j in range(steps_per_epoch):
                """ Discriminator training """
                # get some real episodes
                r_res = self.get_episodes_from_dataset(n=half_batch, random=True)
                r_position, r_env_state, r_action, r_reward, r_next_position, r_next_env_state, r_done = r_res
                # get some fake episodes (env_state and action are still real but can be ones not present in dataset)

                # f_res = self.generate_random_episodes(get_obj=False, n=half_batch)
                # f_position, f_env_state, f_action, _, _, _, _ = f_res
                f_position, f_env_state, f_action = self.generate_new_states(n=half_batch)
                pred_next_env_state, pred_reward, reconstructed_action, latent_space = self.encoder.predict(
                    [f_position, f_env_state, f_action]
                )
                real_labels, fake_labels = (np.ones(shape=(half_batch, 1)), np.zeros(shape=(half_batch, 1)))
                x_discriminator = [
                    np.array([*r_position, *f_position]),
                    np.array([*r_env_state, *f_env_state]),
                    np.array([*r_next_env_state, *pred_next_env_state]),
                    np.array([*r_reward, *pred_reward]),
                    np.array([*r_action, *reconstructed_action]),
                ]
                labels = np.vstack((real_labels, fake_labels))
                # One-sided label smoothing (not sure if it makes sense in rl setting)
                # labels[:half_batch] = 0.9
                # update discriminator model weights
                d_loss, _ = self.enc_discriminator.train_on_batch(x_discriminator, labels)

                """ Autoencoder training """
                # this might result in discriminator outperforming the encoder depending on architecture or vice versa
                if train_encoder:
                    inputs = [r_position, r_env_state, r_action]
                    outputs = [r_next_env_state, r_reward, r_action, inputs]
                    self.autoencoder.train_on_batch(inputs, outputs)

                """ Generator training (discriminator weights deactivated!) """
                # prepare points in latent space as input for the generator
                # f_res = self.generate_new_states(n=half_batch)
                # position, env_state, action, _, _, _, _ = self.generate_random_episodes(get_obj=False, n=batch_size)
                f_position, f_env_state, f_action = self.generate_new_states(n=batch_size)
                # create inverted labels for the fake samples (because generator goal is to trick the discriminator)
                # so our objective (label) is 1 and if discriminator says 1 we have an error of 0 and vice versa
                real_labels = np.ones((batch_size, 1))
                # update the encoder via the discriminator's error
                g_loss = self.enc_gan.train_on_batch([f_position, f_env_state, f_action], real_labels)

                # summarize loss on this batch
                print(
                    f">{i + 1}, {j + 1:0{len(str(steps_per_epoch))}d}/{steps_per_epoch}, d={d_loss:.3f}, g={g_loss:.3f}"
                )
            self.summarize_performance()

    def summarize_performance(self, n=100):
        # get some real episodes
        r_res = self.get_episodes_from_dataset(n=n, random=True)
        r_position, r_env_state, r_action, r_reward, r_next_position, r_next_env_state, r_done = r_res
        # get some fake episodes (env_state and action are still real but can be ones not present in dataset)
        # f_res = self.generate_random_episodes(get_obj=False, n=n)
        # f_position, f_env_state, f_action, _, _, _, _ = f_res
        f_position, f_env_state, f_action = self.generate_new_states(n=n)
        pred_next_env_state, pred_reward, reconstructed_action, latent_space = self.encoder.predict(
            [f_position, f_env_state, f_action]
        )

        real_inputs = [r_position, r_env_state, r_next_env_state, r_reward, r_action]
        fake_inputs = [f_position, f_env_state, pred_next_env_state, pred_reward, reconstructed_action]
        # evaluate discriminator on real examples
        _, acc_real = self.enc_discriminator.evaluate(real_inputs, np.ones(shape=(n, 1)), verbose=0)
        # evaluate discriminator on fake examples
        _, acc_fake = self.enc_discriminator.evaluate(fake_inputs, np.zeros(shape=(n, 1)), verbose=0)
        # summarize discriminator performance
        print(f">Accuracy real: {acc_real * 100:.0f}%, fake: {acc_fake * 100:.0f}%")

    ####################################################################################################################
    """ Methods used for visualization """
    ####################################################################################################################

    def save_model_architecture_images(
        self, models: Optional[List[Model]] = None, path: str = "images/epn_rl/architecture", fmt: str = "png"
    ):
        models = models if models is not None else []
        models.extend(
            [
                self.encoder,
                self.decoder,
                self.autoencoder,
                self.dec_discriminator,
                self.enc_discriminator,
                self.dec_gan,
                self.enc_gan,
            ]
        )
        super().save_model_architecture_images(models, path, fmt)

    def _generate_random_episode(self, get_obj):
        """Generates a random episode consisting of (env_state, action, reward, next_env_state, done).

        :param get_obj: bool
            Whether to return a copy of the maze object in a specific env state or the env state (without walls) itself.

        :return position:
            The one-hot encoded position of the agent.
        :return env_state: Maze
            Either the current env state or a maze object containing the env_state in different formats (rgb, ...).
        :return action: int
            Randomly chosen action between 0 and the number of possible actions - 1 (one-hot encoded).
        :return reward: int
            Received reward from the environment for taking 'action' in 'env_state'.
        :next_env_state: [bool]
            Either the current env state or a maze object containing the next env_state in different formats (rgb, ...).
            The next_env_state is the state received by the environment after taking 'action' in 'env_state'.
        :done: bool
            Information whether the goal was reached by taking 'action' in 'env_state' or not.
        """
        action = np.random.randint(0, self.env.action_space.n)
        self.env.reset(randomize_start=True)
        position = [obj for obj in self.env.maze.objects if obj.name == "agent"][0].positions[0]
        env_state = deepcopy(self.env.maze) if get_obj else self.env.maze.to_valid_obs()
        next_env_state, reward, done, _ = self.env.step(action)
        next_env_state = deepcopy(self.env.maze) if get_obj else self.env.maze.to_valid_obs()
        next_position = [obj for obj in self.env.maze.objects if obj.name == "agent"][0].positions[0]

        # Convert position, next_position and action to one hot
        action_one_hot = tf.keras.utils.to_categorical(action, num_classes=self.env.action_space.n)
        position_1_one_hot = tf.keras.utils.to_categorical(position[0], num_classes=self.env.maze.size[0])
        position_2_one_hot = tf.keras.utils.to_categorical(position[1], num_classes=self.env.maze.size[1])
        position_one_hot = np.concatenate((position_1_one_hot, position_2_one_hot))
        next_position_1_one_hot = tf.keras.utils.to_categorical(next_position[0], num_classes=self.env.maze.size[0])
        next_position_2_one_hot = tf.keras.utils.to_categorical(next_position[1], num_classes=self.env.maze.size[1])
        next_position_one_hot = np.concatenate((next_position_1_one_hot, next_position_2_one_hot))

        return position_one_hot, env_state, action_one_hot, reward, next_position_one_hot, next_env_state, done

    def generate_random_episodes(self, get_obj, n):
        if get_obj:
            env_state = np.empty((n, 1), dtype=type(self.env))
            next_env_state = np.empty((n, 1), dtype=type(self.env))
        else:
            env_state = np.zeros((n, self.nr_valid_tiles), dtype=int)
            next_env_state = np.zeros((n, self.nr_valid_tiles), dtype=int)

        action = np.empty((n, self.env.action_space.n), dtype=int)
        reward = np.empty((n, 1), dtype=float)
        done = np.empty((n, 1), dtype=bool)
        position = np.empty((n, self.env.maze.size[0] + self.env.maze.size[1]), dtype=int)
        next_position = np.empty((n, self.env.maze.size[0] + self.env.maze.size[1]), dtype=int)

        for i in range(n):
            (
                position[i],
                env_state[i],
                action[i],
                reward[i],
                next_position[i],
                next_env_state[i],
                done[i],
            ) = self._generate_random_episode(get_obj)
        return position, env_state, action, reward, next_position, next_env_state, done

    def predict_next_env_state_and_latent_space(self, env_state, action):
        """Given an env_state, the generator predicts the resulting env_state + latent space"""
        inputs = self.env_state_and_action_to_inputs(env_state, action)
        return self.encoder.predict(inputs)

    def env_state_and_action_to_inputs(self, env_state, action):
        action_one_hot = tf.keras.utils.to_categorical(action, num_classes=self.env.action_space.n)
        inputs = np.concatenate((env_state, action_one_hot), axis=1)
        return inputs

    def visualize_outputs_to_file(self, state, n_samples=9, path="images/epn_rl/plots"):
        width = n_samples
        height = 5
        plt.figure(figsize=(width, height))
        for idx in range(n_samples):
            position, env_state_obj, action, _, _, _, _ = self._generate_random_episode(get_obj=True)
            env_state = env_state_obj.to_valid_obs().reshape(1, -1)
            position = position.reshape(1, -1)
            action = action.reshape(1, -1)
            [
                pred_next_env_state,
                pred_reward,
                reconstructed_action,
                (dec_reconstructed_position, dec_reconstructed_state, dec_reconstructed_action),
            ] = self.autoencoder.predict([position, env_state, action])

            # Sampled maze state + action
            add_subplot(image=env_state_obj.to_rgb(), n_cols=height, n_rows=width, index=1 + idx)
            plt.annotate(self.env.motions._fields[np.argmax(action[0])], xy=(0.25, -0.5), fontsize="medium")

            # Sampled maze state + action with next state prediction values
            add_subplot(image=env_state_obj.to_rgb(), n_cols=height, n_rows=width, index=1 + idx + n_samples)
            plt.annotate(round(float(pred_reward), 2), xy=(0.25, -0.5), fontsize="small")
            annotate_maze(pred_next_env_state[0], env_state_obj)

            # Encoder Action reconstructions
            annotate_action_values(
                n_cols=height,
                n_rows=width,
                index=1 + idx + 2 * n_samples,
                action_names=self.env.motions._fields,
                values=reconstructed_action[0],
            )

            # Reconstructed maze state + action (decoder)
            add_subplot(image=env_state_obj.to_rgb(), n_cols=height, n_rows=width, index=1 + idx + 3 * n_samples)
            annotate_maze(dec_reconstructed_state[0], env_state_obj)
            actions_one_hot = dec_reconstructed_action[0]
            annotate_action_values(
                n_cols=height,
                n_rows=width,
                index=1 + idx + 4 * n_samples,
                action_names=self.env.motions._fields,
                values=actions_one_hot,
            )

        save_plot_as_image(path=path, filename=f"{self.nr_tiles}{state}", dpi=300 + self.nr_tiles * 5)


def annotate_maze(values, env_state_obj):
    is_wall = env_state_obj.to_impassable()
    # The values are from row to row and not column to column that's why we start with y!
    for y in range(env_state_obj.size[0]):
        for x in range(env_state_obj.size[1]):
            # Values are only representing valid states, so just annotate if current state is not a wall
            if not is_wall[y, x]:
                # Show value & remove from array. That way we don't need a counter
                plt.annotate(
                    np.round(values[0], 1),
                    xy=(x - 0.35, y + 0.15),
                    fontsize=5 / ((env_state_obj.size[0] * env_state_obj.size[1]) / 15),
                )
                values = np.delete(values, 0)


def annotate_action_values(n_cols, n_rows, index, action_names, values):
    subplot = plt.subplot(n_cols, n_rows, index)
    subplot.axis("off")
    for idx in range(len(action_names)):
        subplot.text(0, 0.75 - idx / 4, f"{action_names[idx]}: {str(np.round(values[idx], 2))}", fontsize="x-small")
