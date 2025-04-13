
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import utils as bandit_utils
from movielens_bandit_env import RealMovieLensEmbeddingEnv
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step

print("Available GPU:", tf.config.list_physical_devices('GPU'))

env = tf_py_environment.TFPyEnvironment(
    RealMovieLensEmbeddingEnv(num_users=50, num_movies=50, embedding_dim=16)
)

agent = lin_ucb_agent.LinearUCBAgent(
    time_step_spec=env.time_step_spec(),
    action_spec=env.action_spec(),
    variable_collection=None,
    accepts_per_arm_features=False,
    alpha=1.0
)
agent.initialize()

reward_history = []
time_step = env.reset()

for step in range(300):
    raw_env = env.pyenv._envs[0]
    current_rating = raw_env._ratings[raw_env._index]

    action_step = agent.policy.action(time_step) #choose the action
    next_time_step = env.step(action_step.action)

    exp = trajectory.Trajectory(
        step_type=tf.expand_dims(time_step.step_type, axis=1),
        observation=tf.expand_dims(time_step.observation, axis=1),
        action=tf.expand_dims(action_step.action, axis=1),
        policy_info=action_step.info,
        next_step_type=tf.expand_dims(next_time_step.step_type, axis=1),
        reward=tf.expand_dims(next_time_step.reward, axis=1),
        discount=tf.expand_dims(next_time_step.discount, axis=1)
    )
    loss = agent.train(exp)

    reward = next_time_step.reward.numpy()[0]
    reward_history.append(reward)
    time_step = env.reset()

    if (step + 1) % 10 == 0:
        print(f"Step {step + 1:3d} | action: {action_step.action.numpy()[0]} | rating: {current_rating:.1f} | reward: {reward:.2f}")

    if (step + 1) % 25 == 0:
            print(f"Step {step+1:3d} | current reward: {reward:.2f} | accumulated reward: {np.mean(reward_history):.4f}")


window = 20
rolling_reward = np.convolve(reward_history, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(reward_history, label='Instant Reward', alpha=0.3)
plt.plot(range(window - 1, len(reward_history)), rolling_reward, label=f'{window}-step Rolling Avg', linewidth=2)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("LinearUCB on Real MovieLens (Smoothed Reward)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
