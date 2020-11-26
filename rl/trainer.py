import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path


class Agent:
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

    def build_model(self):
        pass


class Trainer:
    """Trainer that trains a Q learning agent using a given environment and stores all encountered state action pairs"""

    def __init__(self, env, seed):
        # Setup maze
        self.env = env
        env.seed(seed)
        self.nr_tiles = np.count_nonzero(env.maze.to_impassable() == 0)
        self.agent = Agent(self.env.action_space)
        # check_env requires stable_baselines3 which requires pytorch - This is a overhead for this single command
        # just comment it in if pytorch and stable_baselines3 is installed
        # check_env(self.env)

        # Reset because when using manim, the init maze in the video is where check_env() left off
        # Not required if check_env is commented out. Can be run anyway to have it for the case that check_env is used
        self.env.reset()

    def train(self, epochs, max_episode_length, is_human_mode, randomize_start=True, save_to_file=False):
        print(f"Start training. Possible memories: {(self.nr_tiles * self.agent.action_space.n) - 4}")
        memories = []
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            game_over = False

            # get initial state (1d flattened canvas)
            self.env.reset(randomize_start)
            state = self.env.maze.to_valid_obs()
            state = state.reshape(1, -1)

            n_steps = 0
            for i in range(max_episode_length):
                self.env.render("human") if is_human_mode else self.env.render("rgb_array")

                # Get next action
                action = self.agent.act()

                # Apply action, get reward and new state
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.env.maze.to_valid_obs()
                next_state = next_state.reshape(1, -1)

                if done or i == max_episode_length - 1:
                    game_over = True

                # Store episode if it wasn't experienced before
                action_one_hot = tf.keras.utils.to_categorical(action, num_classes=self.agent.action_space.n)
                s = state.reshape(self.nr_tiles)
                n_s = next_state.reshape(self.nr_tiles)
                if not any([np.array_equal(s, m[0]) and np.array_equal(action_one_hot, m[1]) for m in memories]):
                    # Reshape from 1, X to X (this is better for storing the data)
                    memories.append([s, action_one_hot, n_s, reward, game_over])

                n_steps += 1
                state = next_state

                if game_over:
                    break

            print(f"Epoch: {epoch:03d}/{epochs-1} | Steps: {n_steps}")

        dt = datetime.datetime.now() - start_time
        seconds = dt.total_seconds()
        t = format_time(seconds)
        print(f"n_epoch: {epochs}, time: {t}")

        if save_to_file:
            print(f"Saving {len(memories)} memories to file.")
            save_memories_to_file(memories, filename=self.env.maze.__str__())
        return seconds


def save_memories_to_file(memories, filename, path="data"):
    Path(path).mkdir(parents=True, exist_ok=True)
    np.save(f"{path}/{filename}", memories)


def format_time(seconds):
    # This is a small utility for printing readable time strings:
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)
