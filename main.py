# Streamlit Web App: Dynamic Pricing Model using Q-Learning

import numpy as np
import gym
from gym import spaces
import random
import streamlit as st
import matplotlib.pyplot as plt

# Custom pricing environment
class PricingEnv(gym.Env):
    def __init__(self):
        super(PricingEnv, self).__init__()
        self.price_levels = [10, 15, 20, 25, 30]
        self.state_space = len(self.price_levels)
        self.action_space = spaces.Discrete(3)
        self.state = random.randint(0, self.state_space - 1)

    def reset(self):
        self.state = random.randint(0, self.state_space - 1)
        return self.state

    def step(self, action):
        if action == 0 and self.state > 0:
            self.state -= 1
        elif action == 2 and self.state < self.state_space - 1:
            self.state += 1

        price = self.price_levels[self.state]
        demand = self.simulate_demand(price)
        revenue = price * demand
        reward = revenue
        done = False
        return self.state, reward, done, {}

    def simulate_demand(self, price):
        base_demand = 100
        fluctuation = random.randint(-5, 5)
        return max(base_demand - 3 * price + fluctuation, 0)

# Streamlit app
st.title("📈 Dynamic Pricing with Q-Learning")

# Training settings
episodes = st.slider("Training Episodes", 1000, 10000, 5000, step=1000)
alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
gamma = st.slider("Discount Factor (γ)", 0.5, 0.99, 0.95)
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01

# Initialize env and Q-table
env = PricingEnv()
q_table = np.zeros((env.state_space, env.action_space.n))
rewards = []

# Train model
with st.spinner("Training in progress..."):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(50):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

# Show results
st.subheader("📊 Training Performance")
fig, ax = plt.subplots()
ax.plot(rewards)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Revenue")
ax.set_title("Q-learning Training Performance")
ax.grid(True)
st.pyplot(fig)

st.subheader("📘 Q-Table")
st.dataframe(q_table)

st.subheader("📌 Recommended Pricing Actions")
action_map = ["Lower", "Stay", "Raise"]
for s, price in enumerate(env.price_levels):
    action = np.argmax(q_table[s])
    st.write(f"💰 Price ₹{price} → **{action_map[action]}**")
