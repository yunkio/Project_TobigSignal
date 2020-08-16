import Junction
import Agent
import numpy as np
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import torch

epoch = 10000
agent = Agent.DQNAgent(state_size=24, action_size=8)
reward_info, step_info, longest_wait = [], [], []

for i in tqdm(range(epoch)):
    env = Junction.Junction()
    r = 0
    l = 0
    for step in range(200):
        state = env.get_state()
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.append_sample(state, action, reward, next_state, done)
        agent.train_model()

        l = max(l, env.get_oldest())
        if done == 0:
            r += reward

        # train 할 때는 이 부분 주석 처리
        # print(env.render())
        # sleep(0.1)
        if done == 1:
            reward_info.append(r / (step + 1))
            step_info.append(step)
            longest_wait.append(l)
            break
    
    # train 할 때는 이 부분 주석 처리
    # env.save_graph("fig1.png")
    if len(agent.memory) == 50000 and i % 200 == 199:
        torch.save(agent.model.state_dict(), "weight/save2.pth")
    if i % 200 == 199:
        plt.plot(range(len(reward_info)), step_info)

        plt.savefig(f"graph/graph{i + 1}.png")
        plt.clf()

        plt.plot(range(len(longest_wait)), longest_wait)

        plt.savefig(f"graph/wait{i + 1}.png")
        plt.clf()



