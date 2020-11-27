from typing import List, Optional
from numpy.random import randint
from epn.network import Network
from epn.helper import add_subplot, save_plot_as_image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import random


class NetworkRL(Network):
    """
    A class implementing the Network architecture in an reinforcement learning setting:
    """

    def __init__(
        self,
        env,
        data,
        generator_loss,
        generator_dims: List[int],
        discriminator_dims: List[int],
        seed: int,
    ):
        """
        :param env: MazeEnv
            MazeEnv object initialized with the same maze that was used to generate the dataset.
        """
        super().__init__(False, generator_dims, discriminator_dims, seed)
        self.nr_valid_tiles = np.count_nonzero(env.maze.to_impassable() == 0)
        self.nr_tiles = env.maze.size[0] * env.maze.size[1]
        self.env = env
        self.classification_dim = self.nr_valid_tiles + self.env.action_space.n + 1
        self.generator_loss = generator_loss
        self.train_data, self.test_data, self.combined_data = data

        # Build Autoencoder
        self.generator = self.build_encoder(
            input_tensors=[
                Input(shape=self.nr_valid_tiles, name="state"),
                Input(shape=self.env.action_space.n, name="action"),
            ],
            output_layers=[
                Dense(self.nr_valid_tiles, activation="softmax", name=f"next_state"),
                Dense(1, activation="tanh", name=f"reward"),
            ],
            model_name="generator",
        )
        self.generator.compile(loss=self.generator_loss, optimizer="adam", metrics=["accuracy", "mse"])

        self.discriminator = self.build_discriminator(
            input_tensors=[
                Input(shape=self.nr_valid_tiles, name="state"),
                Input(shape=self.env.action_space.n, name="action"),
                Input(shape=self.nr_valid_tiles, name="next_state"),
                Input(shape=1, name="reward"),
            ],
            output_layers=[Dense(1, activation="sigmoid", name="real_or_fake")],
            model_name="discriminator",
            use_dropout_layers=False,
        )
        self.discriminator.compile(
            loss=["binary_crossentropy"], optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=["accuracy"]
        )

        # Prevent discriminator weight updates during GAN training
        self.discriminator.trainable = False

        # Create the GAN network
        self.gan = self.build_gan(self.generator, self.discriminator)
        self.gan.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

    def build_gan(
        self, generator, discriminator, ignored_layer_names: Optional[List[str]] = None, model_name: str = "GAN"
    ) -> Model:
        inputs = [Input(shape=tensor.shape[1:], name=tensor.name.split(":")[0]) for tensor in generator.inputs]
        outputs = generator(inputs)
        # Only use generated outputs as discriminator inputs that are not specified in 'ignored_layer_names'
        discriminator_inputs = [inputs[0], inputs[1]]
        discriminator_inputs.extend(
            [output for output in outputs if output.name.split("/")[1] not in ignored_layer_names]
            if ignored_layer_names
            else outputs
        )
        discriminated = discriminator(discriminator_inputs)
        return Model(inputs, discriminated, name=model_name)

    def train_generator(self, **kwargs):
        self.generator.fit(
            [self.train_data["state"], self.train_data["action"]],
            [self.train_data["next_state"], self.train_data["reward"]],
            **kwargs,
        )

    def train(self, epochs: int, batch_size: int, steps_per_epoch: int, train_generator_supervised: bool):
        half_batch = int(batch_size / 2)
        # manually enumerate epochs
        for epoch in range(epochs):
            self.visualize_outputs_to_file(state=f"epoch_{epoch}", test_or_train_data="test")
            self.visualize_outputs_to_file(state=f"epoch_{epoch}", test_or_train_data="train")
            # enumerate batches over the training set
            for step in range(steps_per_epoch):
                """ Discriminator training """
                # get indices that are used to extract samples from the dataset
                train_indices = np.random.randint(0, self.train_data["state"].shape[0], half_batch)
                combined_indices = np.random.randint(0, self.combined_data["state"].shape[0], half_batch)

                inputs = [self.combined_data["state"][combined_indices], self.combined_data["action"][combined_indices]]
                next_state, reward = self.generator.predict(inputs)
                x_discriminator = [
                    np.array([*self.train_data["state"][train_indices], *inputs[0]]),
                    np.array([*self.train_data["action"][train_indices], *inputs[1]]),
                    np.array([*self.train_data["next_state"][train_indices], *next_state]),
                    np.array([*self.train_data["reward"][train_indices], *reward]),
                ]
                labels = np.vstack((np.ones(shape=(half_batch, 1)), np.zeros(shape=(half_batch, 1))))
                # One-sided label smoothing (not sure if it makes sense in rl setting)
                # labels[:half_batch] = 0.9
                # update discriminator model weights
                d_loss, _ = self.discriminator.train_on_batch(x_discriminator, labels)

                """ Autoencoder training """
                # this might result in discriminator outperforming the encoder depending on architecture or vice versa
                if train_generator_supervised:
                    inputs = [self.train_data["state"][train_indices], self.train_data["action"][train_indices]]
                    outputs = [self.train_data["next_state"][train_indices], self.train_data["reward"][train_indices]]
                    self.generator.train_on_batch(inputs, outputs)

                """ Generator training (discriminator weights deactivated!) """
                # get indices that are used to extract samples from the dataset
                combined_indices = np.random.randint(0, self.combined_data["state"].shape[0], batch_size)

                # create inverted labels for the fake samples (because generator goal is to trick the discriminator)
                # so our objective (label) is 1 and if discriminator says 1 we have an error of 0 and vice versa
                labels = np.ones((batch_size, 1))

                # update the generator via the discriminator's error
                inputs = [self.combined_data["state"][combined_indices], self.combined_data["action"][combined_indices]]
                g_loss = self.gan.train_on_batch(inputs, labels)

                # summarize loss on this batch
                print(
                    f">{epoch+1}, {step+1:0{len(str(steps_per_epoch))}d}/{steps_per_epoch}, d={d_loss:.3f}, g={g_loss:.3f}"
                )

            self.summarize_performance()

        self.visualize_outputs_to_file(state=f"epoch_{epochs}", test_or_train_data="test")
        self.visualize_outputs_to_file(state=f"epoch_{epochs}", test_or_train_data="train")

    def summarize_performance(self, n=100):
        # evaluate discriminator on fake examples
        test_indices = np.random.randint(0, self.test_data["state"].shape[0], n)
        gen_inputs = [self.test_data["state"][test_indices], self.test_data["action"][test_indices]]
        next_state, reward = self.generator.predict(gen_inputs)
        _, acc_fake = self.discriminator.evaluate([*gen_inputs, next_state, reward], np.zeros(shape=(n, 1)), verbose=0)

        # evaluate discriminator on real examples
        train_indices = np.random.randint(0, self.train_data["state"].shape[0], n)
        r_inputs = [
            self.train_data["state"][train_indices],
            self.train_data["action"][train_indices],
            self.train_data["next_state"][train_indices],
            self.train_data["reward"][train_indices],
        ]
        _, acc_real = self.discriminator.evaluate(r_inputs, np.ones(shape=(n, 1)), verbose=0)

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
                self.generator,
                self.discriminator,
                self.gan,
            ]
        )
        super().save_model_architecture_images(models, path, fmt)

    def visualize_outputs_to_file(
        self, state, n_samples=9, trajectories=False, test_or_train_data="test", path="images/epn_rl/plots"
    ):
        print(f"Using {'test' if test_or_train_data == 'test' else 'train'} samples for plots!")
        data = self.test_data if test_or_train_data == "test" else self.train_data
        indexes = random.sample(range(data["state"].shape[0]), n_samples)
        width = n_samples
        height = 1
        plt.figure(figsize=(width, height))

        next_state = data["state"][indexes[0]].reshape(1, -1)
        for idx in range(n_samples):
            # evaluate discriminator on fake examples
            gen_inputs = [
                data["state"][indexes[idx]].reshape(1, -1) if not trajectories else next_state,
                data["action"][indexes[idx]].reshape(1, -1),
            ]
            next_state, reward = self.generator.predict(gen_inputs)

            # Sampled maze state + action
            maze_rgb = self.env.maze.to_rgb()
            impassable = self.env.maze.to_impassable()
            counter = 0
            next_state_idx = np.argmax(next_state[0])
            state_idx = np.argmax(gen_inputs[0][0])
            for row in range(maze_rgb.shape[0]):
                for column in range(maze_rgb.shape[1]):
                    if impassable[row, column]:
                        continue
                    else:
                        if state_idx == counter:
                            maze_rgb[row, column] = [51, 153, 255]
                        else:
                            maze_rgb[row, column] = [224, 224, 224]

                        if counter == next_state_idx:
                            # Color the block as red if its the one predicted as next state with highest accuracy
                            maze_rgb[row, column] = [250, 100, 100]
                        counter += 1

            add_subplot(image=maze_rgb, n_cols=height, n_rows=width, index=1 + idx)
            plt.annotate(
                f"{self.env.motions._fields[np.argmax(gen_inputs[1])]}, {round(float(reward), 2)}",
                xy=(0, -0.5),
                fontsize="x-small",
            )
            annotate_maze(next_state[0], self.env)

        save_plot_as_image(
            path=path, filename=f"{self.nr_tiles}_{state}_{test_or_train_data}.png", dpi=300 + self.nr_tiles * 5
        )


def annotate_maze(values, env):
    is_wall = env.maze.to_impassable()
    # The values are from row to row and not column to column that's why we start with y!
    for y in range(env.maze.size[0]):
        for x in range(env.maze.size[1]):
            # Values are only representing valid states, so just annotate if current state is not a wall
            if not is_wall[y, x]:
                # Show value & remove from array. That way we don't need a counter
                plt.annotate(
                    np.round(values[0], 1),
                    xy=(x - 0.35, y + 0.15),
                    fontsize=5 / ((env.maze.size[0] * env.maze.size[1]) / 15),
                )
                values = np.delete(values, 0)
