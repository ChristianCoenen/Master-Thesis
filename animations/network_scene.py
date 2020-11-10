import datetime
import random
from experience import Experience
from manim import *
from animations.network_m_object_simple import NetworkMobjectSimple
from trainer import Trainer, format_time
from maze.predefined_maze import x3
from maze import Maze


class RLScene(Scene):
    CONFIG = {
        "network_mob_config": {
            "neuron_radius": 0.25,
            "bias_radius": 0.25,
            "bias_stroke_color": COLOR_MAP["GRAY"],
            "layer_to_layer_buff": LARGE_BUFF * 1.5,
            "edge_propagation_color": COLOR_MAP["RED_C"],
            "nn_position": UP * 0.8,
            "include_bias": False,
            "neuron_to_neuron_buff": SMALL_BUFF,
        },
        "element_to_mobject_config": {
            "num_decimal_places": 2,
            "include_sign": True,
        },
    }

    def __init__(self, **kwargs):
        # init is not called when running it via manim, but I declared everything here to avoid IDE warnings
        super().__init__(**kwargs)
        self.trainer = None
        self.epochs = None
        self.max_episode_length = None
        self.data_size = None
        self.epsilon = None
        self.memory_size = None

        self.output_labels = None
        self.maze = None
        self.layer_sizes = None
        self.network_mob = None
        self.manim_weight_matrices = None
        self.reward = None
        self.epoch_step_counter = None

    def setup(self):
        self.trainer = Trainer(maze=Maze(x3))
        self.epochs = 1
        self.max_episode_length = 10
        self.data_size = 1
        self.epsilon = 0.1
        # self.memory_size = 8 * self.trainer.nr_tiles
        self.memory_size = 1

        self.output_labels = VGroup(*[Text(name, font="Roboto") for name in self.trainer.env.motions._fields])

        # Initialize the maze
        self.init_maze()

        # Fill the layer sizes and weight matrices
        self.layer_sizes = []
        for layer in self.trainer.model.layers:
            self.layer_sizes.append(layer.input_shape[1]) if hasattr(layer, "input_shape") else None
            self.layer_sizes.append(layer.units) if hasattr(layer, "units") else None

        self.init_weights()

        # Create a neural network based on the layer sizes
        self.network_mob = NetworkMobjectSimple(self.layer_sizes, **self.network_mob_config)

        # Create reward text & set it to zero
        self.reward = VGroup(*[Text("Reward: ", font="Roboto"), DecimalNumber(0.0)])

        # Create epoch / step counter & set it to zero
        self.epoch_step_counter = VGroup(
            VGroup(*[Text("Epoch: ", font="Roboto"), Integer(0), Text(" / ", font="Roboto")], Integer(self.epochs)),
            VGroup(
                *[Text("Step: ", font="Roboto"), Integer(0), Text(" / ", font="Roboto")],
                Integer(self.max_episode_length),
            ),
        )

    ####################################################################################################################
    """ MAZE METHODS """
    ####################################################################################################################
    def get_maze_colors(self):
        colors = []
        maze_values = self.trainer.env.maze.to_value()
        for value in maze_values.flatten():
            # object_index is the index of the object which has the value specified
            object_index = int(np.where([obj.value == value for obj in self.trainer.env.maze.objects])[0])
            rgb_tuple = self.trainer.env.maze.objects[object_index].rgb
            rgb_hex = "#{:02x}{:02x}{:02x}".format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2])
            colors.append(rgb_hex)
        return colors

    def init_maze(self):
        maze_colors = self.get_maze_colors()
        self.maze = VGroup(*[Square(fill_color=color, fill_opacity=1) for color in maze_colors])

    def update_maze(self):
        maze_colors = self.get_maze_colors()
        [self.maze[i].set_fill(color=maze_colors[i]) for i in range(len(self.maze))]

    ####################################################################################################################
    """ WEIGHT METHODS """
    ####################################################################################################################
    def get_weights(self):
        weight_matrices = []
        for idx, layer in enumerate(self.trainer.model.layers):
            normal_weights = None
            bias_weights = None
            if hasattr(layer, "kernel") and layer.kernel is not None:
                normal_weights = layer.kernel.numpy().transpose()
            if hasattr(layer, "bias") and layer.bias is not None:
                bias_weights = layer.bias.numpy()
                bias_weights.shape += (1,)

            if normal_weights is not None and bias_weights is not None:
                weights = np.concatenate((bias_weights, normal_weights), axis=1)
            elif normal_weights is not None:
                weights = normal_weights
            else:
                weights = bias_weights
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

    def visualize_input(self, env_state):
        # Bias neuron is always activated (cause it is always 1)
        if hasattr(self.network_mob.layers[0], "bias") and self.network_mob.layers[0].bias is not None:
            self.network_mob.layers[0].bias.set_fill(color=COLOR_MAP["WHITE"], opacity=1)

        for idx, bool_input in enumerate(env_state.flatten()):
            if bool_input:
                self.network_mob.layers[0].neurons[idx].set_fill(color=COLOR_MAP["WHITE"], opacity=1)
            else:
                self.network_mob.layers[0].neurons[idx].set_fill(opacity=0)

    def highlight_selected_action(self, action, is_random_action):
        for i in range(len(self.output_labels)):
            if i == action:
                self.output_labels[i].set_color(COLOR_MAP["RED_C"] if is_random_action else COLOR_MAP["GREEN_C"])
            else:
                self.output_labels[i].set_color(COLOR_MAP["WHITE"])

    def show_reward(self, reward):
        self.reward[-1].set_value(reward)
        self.reward[-1].set_color(COLOR_MAP["GREEN_C"] if reward >= 0 else COLOR_MAP["RED_C"])

    def train(self):
        self.trainer.train(
            epochs=self.epochs,
            data_size=self.data_size,
            epsilon=self.epsilon,
            max_episode_length=self.max_episode_length,
            max_memory=self.memory_size,
            is_human_mode=False,
            trainer_scene=self,
        )
        start_time = datetime.datetime.now()

        # Initialize experience replay object
        experience = Experience(self.trainer.model, max_memory=self.memory_size)

        # history of win/lose game
        win_history = []
        win_rate = 0.0

        for epoch in range(self.epochs):
            loss = 0.0
            game_over = False

            # get initial env_state (1d flattened canvas)
            env_state = self.trainer.env.reset()
            env_state = self.trainer.env.maze.to_valid_obs() if self.trainer.only_free_tiles else env_state
            env_state = env_state.reshape(1, -1)
            # Show initial env state in maze and as input to the neurons
            self.update_maze()
            self.update_epoch_step_counter(epoch=epoch + 1, step=0)
            self.visualize_input(env_state)

            n_steps = 0
            for i in range(self.max_episode_length):
                self.trainer.env.render("rgb_array")

                prev_env_state = env_state
                # Get next action
                outputs = experience.predict(prev_env_state)
                if np.random.rand() < self.epsilon:
                    action = random.choice(range(self.trainer.env.action_space.n))
                    is_random_action = True
                else:
                    action = np.argmax(experience.predict(prev_env_state))
                    is_random_action = False

                self.network_mob.add_output_values(outputs, scale_factor=0.3)

                # Apply action, get reward and new env_state
                env_state, reward, done, _ = self.trainer.env.step(action)
                env_state = self.trainer.env.maze.to_valid_obs() if self.trainer.only_free_tiles else None
                env_state = env_state.reshape(1, -1)

                self.highlight_selected_action(action, is_random_action=is_random_action)
                self.show_reward(reward)

                if done:
                    win_history.append(1)
                    game_over = True
                elif i == self.max_episode_length - 1:
                    win_history.append(0)
                    game_over = True

                # Store episode (experience)
                episode = [prev_env_state, action, reward, env_state, game_over]
                experience.remember(episode)
                n_steps += 1

                # Train neural network model
                inputs, targets = experience.get_data(data_size=self.data_size)
                self.trainer.model.fit(inputs, targets, epochs=1, batch_size=1, verbose=0)
                loss = self.trainer.model.evaluate(inputs, targets, verbose=0)

                self.update_weights()
                self.wait(1)
                self.update_maze()
                self.update_epoch_step_counter(epoch=epoch + 1, step=i + 1)
                self.visualize_input(env_state)

                if game_over:
                    break

            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
            print(
                f"Epoch: {epoch:03d}/{self.epochs - 1} | Loss: {loss:.4f} | Steps: {n_steps} | "
                f"Win count: {sum(win_history)} | Win rate: {win_rate:.3f} | time: {t}"
            )

        dt = datetime.datetime.now() - start_time
        seconds = dt.total_seconds()
        t = format_time(seconds)
        print(f"n_epoch: {self.epochs}, max_mem: {self.memory_size}, data: {self.data_size}, time: {t}")
        return seconds

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
        # Add maze
        self.maze.scale(0.35)
        self.maze.arrange_in_grid(3, 3, buff=0)
        self.maze.next_to(self.network_mob, LEFT, buff=LARGE_BUFF)
        self.add(self.maze)

        # Add neural network
        self.add(self.network_mob)

        # Add output labels next to the output neurons
        self.output_labels.scale(0.4)
        self.output_labels.arrange(DOWN)
        for i in range(len(self.output_labels)):
            self.output_labels[i].next_to(self.network_mob.layers[-1].neurons[i], RIGHT)
        self.add(self.output_labels)

        # Add weight matrices
        self.manim_weight_matrices.scale(0.35)
        self.manim_weight_matrices.arrange(RIGHT)
        self.manim_weight_matrices.next_to(self.network_mob, DOWN)
        self.add(self.manim_weight_matrices)

        # Add reward object next to the output labels
        self.reward.scale(0.5)
        self.reward.arrange(RIGHT, buff=0)
        self.reward.next_to(self.output_labels, RIGHT)
        self.add(self.reward)

        # Add epoch / step counter object in the upper left corner
        self.epoch_step_counter.scale(0.5)
        [line.arrange(RIGHT, buff=0) for line in self.epoch_step_counter]
        self.epoch_step_counter.arrange(DOWN, aligned_edge=RIGHT)
        self.epoch_step_counter.shift(UP * 3 + RIGHT * 4)
        self.add(self.epoch_step_counter)

        self.wait(1)
        self.train()
        self.wait(1)
