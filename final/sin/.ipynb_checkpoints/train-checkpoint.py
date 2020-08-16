import Junction
import Agent
import numpy as np
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import torch

train = False
epoch = 10000 if train else 1
agent = Agent.DQNAgent(state_size=24, action_size=8, train=train, state_dict=torch.load("weight/save.pth"))
reward_info, step_info, longest_wait = [], [], []
avg_reward, avg_step = 0, 0

for i in tqdm(range(epoch)):
    env = Junction.Junction()
    r, l = 0, 0
    for step in range(200):
        state = env.get_state()
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        if train:
            agent.append_sample(state, action, reward, next_state, done)
            agent.train_model()

        l = max(l, env.get_oldest())
        r += reward

        # train 할 때는 이 부분 주석 처리
        if not train:
            print(env.render())
            sleep(0.1)

        if done == 1:
            break

    if not train:
        env.save_graph("fig.png")
    if train:
        if avg_reward == 0:
            avg_reward = r
        else:
            avg_reward = avg_reward * 0.8 + r * 0.2
        if avg_step == 0:
            avg_step = step
        else:
            avg_step = avg_step * 0.8 + step * 0.2

        if i % 10 == 9:
            reward_info.append(avg_reward)
            step_info.append(avg_step)
            longest_wait.append(l)
        
        
        if len(agent.memory) == 10000 and i % 200 == 199:
            torch.save(agent.model.state_dict(), "weight/save.pth")
        if i % 500 == 499:
            plt.plot(range(len(reward_info)), step_info)

            plt.savefig(f"graph/graph{i + 1}.png")
            plt.clf()

            plt.plot(range(len(longest_wait)), longest_wait)

            plt.savefig(f"graph/wait{i + 1}.png")
            plt.clf()

            plt.plot(range(len(reward_info)), step_info)

            plt.savefig(f"graph/graph{i + 1}.png")
            plt.clf()

            plt.plot(range(len(reward_info)), reward_info)

            plt.savefig(f"graph/avgreward{i + 1}.png")
            plt.clf()



