import gym
# load CartPole env version 1
env = gym.make("CartPole-v1",  render_mode="rgb_array")
def basic_policy(obs):
    if len(obs) == 2:
       angle = obs[0][2]
    else : 
       angle = obs[2]
    return 0 if angle < 0 else 1
totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(list(obs))
        obs, reward, done, info, _ = env.step(action)
        episode_rewards += reward
        env.render()
        if done :
            break
    totals.append(episode_rewards)

import numpy as np
print(f"mean: {np.mean(totals)},\n std: {np.std(totals)} \n min: {np.min(totals)} \n max: {np.max(totals)}")