import Junction
import Agent
import numpy as np
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import torch

epoch = 10000
agent = Agent.DQNAgent(action_size=8)
reward_info, step_info = [], []

for i in tqdm(range(epoch)):
    env = Junction.Junction()
    r = 0
    for step in range(200):
        state = env.get_state()
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.append_sample(state, action, reward, next_state, done)
        agent.train_model()

        if done == 0:
            r += reward

        # train 할 때는 이 부분 주석 처리
        # print(env.render())
        # sleep(0.1)
        if done == 1:
            reward_info.append(r / (step + 1))
            step_info.append(step)
            break
    
    # train 할 때는 이 부분 주석 처리
    # env.save_graph("fig1.png")
    if len(agent.memory) == 10000:
        torch.save(agent.model.state_dict(), "weight/save.pth")
    if i % 100 == 99:
        plt.plot(range(len(reward_info)), step_info)
        plt.legend()

        plt.savefig(f"graph/graph{i + 1}.png")



