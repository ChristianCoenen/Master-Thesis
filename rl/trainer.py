import datetime
import numpy as np
import random
import tensorflow as tf
from rl.experience import Experience


class Trainer:
    def __init__(self, env, only_free_tiles=True):
        # Setup maze
        self.only_free_tiles = only_free_tiles
        self.env = env
        self.nr_tiles = self.nr_tiles = (
            np.count_nonzero(env.maze.to_impassable() == 0) if only_free_tiles else env.maze.size[0] * env.maze.size[1]
        )
        # check_env requires stable_baselines3 which requires pytorch - This is a overhead for this single command
        # just comment it in if pytorch and stable_baselines3 is installed
        # check_env(self.env)

        # Reset because when using manim, the init maze in the video is where check_env() left off
        # Not required if check_env is commented out. Can be run anyway to have it for the case that check_env is used
        self.env.reset()

        self.model = build_model(self.env, self.nr_tiles)

    def train(
        self,
        epochs,
        data_size,
        epsilon,
        max_episode_length,
        max_memory,
        is_human_mode,
        save_to_file=False,
    ):
        start_time = datetime.datetime.now()

        # Initialize experience replay object
        experience = Experience(self.model, max_memory=max_memory)

        # history of win/lose game
        win_history = []
        win_rate = 0.0

        for epoch in range(epochs):
            loss = 0.0
            game_over = False

            # get initial env_state (1d flattened canvas)
            env_state = self.env.reset()
            env_state = self.env.maze.to_valid_obs() if self.only_free_tiles else env_state
            env_state = env_state.reshape(1, -1)
            position = [obj for obj in self.env.maze.objects if obj.name == "agent"][0].positions[0]

            n_steps = 0
            for i in range(max_episode_length):
                if is_human_mode:
                    self.env.render("human")
                else:
                    self.env.render("rgb_array")

                prev_env_state = env_state
                prev_position = position
                # Get next action

                if np.random.rand() < epsilon:
                    action = random.choice(range(self.env.action_space.n))
                else:
                    action = np.argmax(experience.predict(prev_env_state))

                # Apply action, get reward and new env_state
                env_state, reward, done, _ = self.env.step(action)
                env_state = self.env.maze.to_valid_obs() if self.only_free_tiles else None
                env_state = env_state.reshape(1, -1)
                position = [obj for obj in self.env.maze.objects if obj.name == "agent"][0].positions[0]

                if done:
                    win_history.append(1)
                    game_over = True
                elif i == max_episode_length - 1:
                    win_history.append(0)
                    game_over = True

                # Store episode (experience)
                episode = [prev_position, prev_env_state, action, reward, position, env_state, game_over]
                experience.remember(episode)
                n_steps += 1

                # Train neural network model
                inputs, targets = experience.get_data(data_size=data_size)
                self.model.fit(inputs, targets, epochs=1, batch_size=1, verbose=0)
                loss = self.model.evaluate(inputs, targets, verbose=0)

                if game_over:
                    break

            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
            print(
                f"Epoch: {epoch:03d}/{epochs-1} | Loss: {loss:.4f} | Steps: {n_steps} | "
                f"Win count: {sum(win_history)} | Win rate: {win_rate:.3f} | time: {t}"
            )

            if win_rate > 0.9:
                epsilon = 0.05

        dt = datetime.datetime.now() - start_time
        seconds = dt.total_seconds()
        t = format_time(seconds)
        print(f"n_epoch: {epochs}, max_mem: {max_memory}, data: {data_size}, time: {t}")
        if save_to_file:
            print("Saving memory to file!")
            experience.save_memories_to_file(maze=self.env.maze, filename=self.env.maze.__str__())
        return seconds


def build_model(env, nr_tiles):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(env.action_space.n, input_shape=(nr_tiles,), kernel_initializer="zeros", use_bias=False)
    )
    # model.add(tf.keras.layers.PReLU())
    sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False, name="SGD")
    model.compile(optimizer=sgd, loss="mse")
    return model


# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)
