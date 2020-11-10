import tensorflow as tf
from manim import *
from animations.network_m_object_simple import NetworkMobjectSimple
from epn.network import EntropyPropagationNetwork


class RLScene(Scene):
    CONFIG = {
        "network_mob_config": {
            "neuron_radius": 0.15,
            "bias_radius": 0.15,
            "bias_stroke_color": COLOR_MAP["GRAY"],
            "layer_to_layer_buff": LARGE_BUFF * 3,
            "edge_propagation_color": COLOR_MAP["RED_C"],
            "nn_position": UP * 1.2,
            "include_bias": True,
            "neuron_to_neuron_buff": 0.05,
            "edge_stroke_width": 1,
        },
        "element_to_mobject_config": {
            "num_decimal_places": 2,
            "include_sign": True,
        },
    }

    def __init__(self, **kwargs):
        # init is not called when running it via manim, but I declared everything here to avoid IDE warnings
        super().__init__(**kwargs)
        self.encoder_dims = None
        self.latent_dim = None
        self.classification_dim = None
        self.epn = None
        self.epochs = None
        self.batch_size = None
        self.action_names = None

        self.output_labels = None
        self.maze = None
        self.layer_sizes = None
        self.network_mob = None
        self.manim_weight_matrices = None
        self.epoch_step_counter = None

    def setup(self):
        self.epochs = 1
        self.batch_size = 1
        self.action_names = ["north", "south", "west", "east"]
        self.encoder_dims = [12]
        self.latent_dim = 3
        self.classification_dim = 9
        self.epn = EntropyPropagationNetwork(
            dataset="maze_memories",
            dataset_path="/Users/christiancoenen/repos/Thesis/Thesis-Reinforcement-Learning/data/9tiles_maze_memories.npy",
            shuffle_data=False,
            encoder_dims=self.encoder_dims,
            latent_dim=self.latent_dim,
            classification_dim=self.classification_dim,
            weight_sharing=True,
            graphviz_installed=True,
        )

        # Fill the layer sizes and weight matrices
        self.layer_sizes = [
            self.epn.x_train_norm.shape[1],
            *[encoder_dim for encoder_dim in self.encoder_dims],
            self.latent_dim + self.classification_dim,
        ]

        self.init_weights()

        # Create a neural network based on the layer sizes
        self.network_mob = NetworkMobjectSimple(self.layer_sizes, **self.network_mob_config)

        # Create epoch / step counter & set it to zero
        self.epoch_step_counter = VGroup(
            VGroup(*[Text("Epoch: ", font="Roboto"), Integer(0), Text(" / ", font="Roboto")], Integer(self.epochs)),
            VGroup(
                *[Text("Step: ", font="Roboto"), Integer(0), Text(" / ", font="Roboto")],
                Integer(int(np.ceil(self.epn.x_train_norm.shape[0] / self.batch_size))),
            ),
        )

    ####################################################################################################################
    """ WEIGHT METHODS """

    ####################################################################################################################
    def get_weights(self):
        weight_matrices = []
        for idx, layer in enumerate(self.epn.encoder.layers):
            normal_weights = None
            bias_weights = None
            if hasattr(layer, "kernel") and layer.kernel is not None:
                normal_weights = layer.kernel.numpy().transpose()
            if hasattr(layer, "bias") and layer.bias is not None:
                bias_weights = layer.bias.numpy()
                bias_weights.shape += (1,)
            if type(self.epn.encoder.layers[idx]).__name__ == "Concatenate":
                # This code snippet assumes that the last layers are concatenated. It needs adjustment otherwise
                concat_weights = []
                for i in range(len(layer.inbound_nodes)):
                    concat_weights.append(weight_matrices[-1])
                    weight_matrices.remove(weight_matrices[-1])
                weight_matrices.append(np.concatenate(concat_weights))

            if normal_weights is not None and bias_weights is not None:
                weights = np.concatenate((bias_weights, normal_weights), axis=1)
            elif normal_weights is not None:
                weights = normal_weights
            elif bias_weights is not None:
                weights = bias_weights
            else:
                continue
            weight_matrices.append(weights)

        return weight_matrices

    def init_weights(self):
        weight_matrices = self.get_weights()
        self.manim_weight_matrices = VGroup(
            *[
                DecimalMatrix(weight_matrix, element_to_mobject_config=self.element_to_mobject_config, h_buff=1.7)
                for weight_matrix in weight_matrices
            ]
        )

    def update_weights(self):
        weight_matrices = self.get_weights()

        for m_idx, old_weight_matrix in enumerate(self.manim_weight_matrices):
            new_weight_matrix = weight_matrices[m_idx].flatten()
            for n_idx, number in enumerate(old_weight_matrix[0]):
                if number.get_value() > new_weight_matrix[n_idx]:
                    number.set_color(color=COLOR_MAP["RED_C"])
                elif number.get_value() < new_weight_matrix[n_idx]:
                    number.set_color(color=COLOR_MAP["GREEN_C"])
                else:
                    number.set_color(color=COLOR_MAP["WHITE"])
                number.set_value(new_weight_matrix[n_idx])

    ####################################################################################################################
    """ NETWORK METHODS """

    ####################################################################################################################
    def feed_network(self, forward=True, width=1.0):
        layer_order = range(len(self.layer_sizes))
        if not forward:
            layer_order = reversed(layer_order)

        for i in layer_order:
            self.show_activation_of_layer(i, forward, width)

    def show_activation_of_layer(self, layer_index, forward, width):
        if layer_index > 0:
            anim = self.network_mob.get_edge_propagation_animations(
                layer_index - 1, lag_ratio=0, forward=forward, width=width
            )
            self.play(*anim)

    def visualize_data(self, values, layer_idx):
        # Bias neuron is always activated (cause it is always 1)
        if hasattr(self.network_mob.layers[layer_idx], "bias") and self.network_mob.layers[layer_idx].bias is not None:
            self.network_mob.layers[layer_idx].bias.set_fill(color=COLOR_MAP["WHITE"], opacity=1)

        for idx, value in enumerate(values.flatten()):
            self.network_mob.layers[layer_idx].neurons[idx].set_fill(color=COLOR_MAP["WHITE"], opacity=value)

    def visualize_input(self, values):
        self.visualize_data(values, 0)

    def visualize_predictions(self, values):
        self.visualize_data(values, len(self.layer_sizes) - 1)

    def visualize_reconstructions(self, values):
        # TODO: decoder is not visualized (don't know if it makes sense to visualize it)
        pass

    def train(self):
        for epoch in range(self.epochs):
            # TODO: currently only working for batch size 1 (might not be fixed cause 1 is best for visualization)
            for idx, (x_train_norm, y_train) in enumerate(zip(self.epn.x_train_norm, self.epn.y_train)):
                self.update_epoch_step_counter(epoch + 1, idx + 1)
                # Transform vectors to matrices - otherwise keras can't process them
                x_train_norm = x_train_norm.reshape(1, -1)
                y_train = y_train.reshape(1, -1)

                # Calculate outputs and visualize both inputs and outputs
                # TODO: also get hidden layer outputs
                outputs = self.epn.autoencoder.predict(x_train_norm)
                self.visualize_input(x_train_norm)
                self.visualize_predictions(outputs[0][0])
                self.visualize_reconstructions(outputs[1][0])

                # Train the network on the visualized inputs / outputs and visualize the updated weights
                self.epn.autoencoder.train_on_batch(x_train_norm, [y_train, x_train_norm])
                self.update_weights()
                if idx > 5:
                    break
                self.wait(1)

    ####################################################################################################################
    """ OTHER METHODS """

    ####################################################################################################################
    def update_epoch_step_counter(self, epoch, step):
        self.epoch_step_counter[0][1].set_value(epoch)
        self.epoch_step_counter[1][1].set_value(step)

    ####################################################################################################################
    """ DRAW METHOD """

    ####################################################################################################################
    def construct(self):
        # Add neural network
        self.add(self.network_mob)

        # Add weight matrices
        self.manim_weight_matrices.scale(0.2)
        self.manim_weight_matrices.arrange(RIGHT)
        self.manim_weight_matrices.next_to(self.network_mob, DOWN)
        self.add(self.manim_weight_matrices)

        # Add epoch / step counter object in the upper left corner
        self.epoch_step_counter.scale(0.5)
        [line.arrange(RIGHT, buff=0) for line in self.epoch_step_counter]
        self.epoch_step_counter.arrange(DOWN, aligned_edge=RIGHT)
        self.epoch_step_counter.shift(UP * 3 + RIGHT * 5)
        self.add(self.epoch_step_counter)

        self.train()
        self.wait(1)
