import gym
import rl2.trainer as trainer

env = gym.make("CarRacing-v0")
trainer.train(env, epochs=10, is_human_mode=False, save_to_file=True)
