import sys
import os
sys.path.insert(0, os.getcwd())
import network.helper as helper
import gym
from manim import *
from manim import color as c
from animation.network_m_object_simple import NetworkMobjectSimple
from rl.trainer import Trainer
from maze.predefined_maze import x3
from maze import Maze

USE_BIAS = False


class RLScene(Scene):
    CONFIG = {
        "network_mob_config": {
            "neuron_radius": 0.25,
            "bias_radius": 0.25,
            "bias_stroke_color": c.GRAY,
            "layer_to_layer_buff": LARGE_BUFF * 1.5,
            "edge_propagation_color": c.RED,
            "nn_position": UP * 0.8,
            "include_bias": USE_BIAS,
            "neuron_to_neuron_buff": SMALL_BUFF,
        },
        "element_to_mobject_config": {
            "num_decimal_places": 2,
            "include_sign": True,
        },
    }

    def setup(self):
        seed_value = 30
        helper.set_seeds(seed_value)

        self.env = gym.make("maze:Maze-v0", maze=Maze(x3))
        self.trainer = Trainer(self.env, seed_value)
        self.trainer.agent.model = self.trainer.agent.build_model(bias=USE_BIAS)
        self.epochs = 10
        self.max_episode_length = 25
        self.epsilon = 0.1

        self.output_labels = VGroup(*[Text(name, font="Roboto") for name in self.env.motions._fields])

        # Initialize the maze
        self.init_maze()

        # Fill the layer sizes and weight matrices
        self.layer_sizes = []
        for layer in self.trainer.agent.model.layers:
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
        maze_values = self.env.maze.to_value()
        for value in maze_values.flatten():
            # object_index is the index of the object which has the value specified
            object_index = int(np.where([obj.value == value for obj in self.env.maze.objects])[0])
            rgb_tuple = self.env.maze.objects[object_index].rgb
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
        for idx, layer in enumerate(self.trainer.agent.model.layers):
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
                    number.set_color(color=c.RED)
                elif number.get_value() < new_weight_matrix[n_idx]:
                    number.set_color(color=c.GREEN)
                else:
                    number.set_color(color=c.WHITE)
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
            self.network_mob.layers[0].bias.set_fill(color=c.WHITE, opacity=1)

        for idx, bool_input in enumerate(env_state.flatten()):
            if bool_input:
                self.network_mob.layers[0].neurons[idx].set_fill(color=c.WHITE, opacity=1)
            else:
                self.network_mob.layers[0].neurons[idx].set_fill(opacity=0)

    def highlight_selected_action(self, action, is_random_action):
        for i in range(len(self.output_labels)):
            if i == action:
                self.output_labels[i].set_color(c.RED if is_random_action else c.GREEN)
            else:
                self.output_labels[i].set_color(c.WHITE)

    def show_reward(self, reward):
        self.reward[-1].set_value(reward)
        self.reward[-1].set_color(c.GREEN if reward >= 0 else c.RED)

    def train(self):
        for epoch in range(self.epochs):
            game_over = False

            # get initial state (1d flattened canvas)
            self.env.reset(randomize_start=False)
            state = self.env.maze.to_valid_obs()
            state = state.reshape(1, -1)
            # Show initial env state in maze and as input to the neurons
            self.update_maze()
            self.update_epoch_step_counter(epoch=epoch + 1, step=0)
            self.visualize_input(state)

            n_steps = 0
            for i in range(self.max_episode_length):
                self.env.render("rgb_array")

                # Get next action
                if np.random.rand() < self.epsilon:
                    action = self.trainer.agent.act_random()
                    is_random_action = True
                else:
                    action = np.argmax(self.trainer.agent.act(state))
                    is_random_action = False

                q_values = self.trainer.agent.act(state)
                self.network_mob.add_output_values(q_values, scale_factor=0.3)

                # Apply action, get reward and new state
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.env.maze.to_valid_obs()
                next_state = next_state.reshape(1, -1)

                self.highlight_selected_action(action, is_random_action=is_random_action)
                self.show_reward(reward)

                if done or i == self.max_episode_length - 1:
                    game_over = True

                # Train neural network model
                target_q_values = self.trainer.agent.get_target_q_values(state, action, next_state, reward, done)
                self.trainer.agent.model.fit(state, target_q_values, epochs=1, batch_size=1, verbose=0)

                n_steps += 1
                state = next_state

                self.update_weights()
                self.wait(1)
                self.update_maze()
                self.update_epoch_step_counter(epoch=epoch + 1, step=i + 1)
                self.visualize_input(state)

                if game_over:
                    break

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
        self.reward.arrange(RIGHT, buff=SMALL_BUFF)
        self.reward.next_to(self.output_labels, RIGHT)
        self.add(self.reward)

        # Add epoch / step counter object in the upper left corner
        self.epoch_step_counter.scale(0.5)
        [line.arrange(RIGHT, buff=SMALL_BUFF) for line in self.epoch_step_counter]
        self.epoch_step_counter.arrange(DOWN, aligned_edge=RIGHT)
        self.epoch_step_counter.shift(UP * 3 + RIGHT * 4)
        self.add(self.epoch_step_counter)

        self.wait(1)
        self.train()
        self.wait(1)
