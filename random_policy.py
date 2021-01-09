import gym
import numpy as np
env = gym.make("Seaquest-v0")

rewards = []
for i_epi in range(5):
    episode_rewards = 0
    state = env.reset()
    t = 0
    steps = []
    while True:
        t += 1
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        episode_rewards += reward
        if i_epi == 0:
            env.render()
        if done:
            env.close()
            steps.append(t)
            rewards.append(episode_rewards)
            break

mean_step = np.mean(steps)
mean_reward = np.mean(rewards)
print("average steps", mean_step)
print("average reward ", mean_reward)
