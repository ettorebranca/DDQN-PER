import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque, namedtuple 


class QNetwork(nn.Module):
    def __init__(self, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(96, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(96, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size) 
        
    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def sample(self, beta=0.1):
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()
        sample_indices = np.random.choice(len(self.memory), self.batch_size, p=sampling_probabilities)
        experiences = [self.memory[idx] for idx in sample_indices]
        weights = (len(self.memory) * sampling_probabilities[sample_indices]) ** (-beta)    
        weights /= weights.max()
        states = np.array([e.state for e in experiences], dtype=np.float32) 
        actions = np.array([e.action for e in experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32) 
        dones = np.array([e.done for e in experiences], dtype=np.float32)
        states = torch.tensor(states).float().permute(0, 3, 1, 2)
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(next_states).float().permute(0, 3, 1, 2)
        dones = torch.tensor(dones).float()
        return (states, actions, rewards, next_states, dones, weights, sample_indices)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  

    def __len__(self):
        return len(self.memory)


class Policy(nn.Module):
    continuous = False  
    def __init__(self, state_size=(96, 96, 3), action_size=5, seed=0, device=torch.device('cpu'), 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99, beta_start=0.1, beta_end=1.0, beta_frames=250000):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = device
        self.epsilon = epsilon_start 
        self.epsilon_end = epsilon_end 
        self.epsilon_decay = epsilon_decay
        self.beta_start = beta_start 
        self.beta_end = beta_end 
        self.beta_frames = beta_frames 
        self.beta = beta_start
        self.qnetwork_local = QNetwork(action_size, seed).to(device)
        self.qnetwork_target = QNetwork(action_size, seed).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.memory = PrioritizedReplayBuffer(buffer_size=300000, batch_size=64, alpha=0.6)
        self.t_step = 0

    def forward(self, x):
        return self.qnetwork_local(x)

    def act(self, state): 
        state = torch.tensor(state / 255.0).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)   
        with torch.no_grad():
            action_values = self.qnetwork_local(state)      
        if np.random.rand() > self.epsilon:
            action = action_values.argmax(dim=1).item()  
        else:
            action = np.random.choice(self.action_size) 
        return action

    def learn(self, experiences, gamma, weights):
        states, actions, rewards, next_states, dones, weights, sample_indices = experiences
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        best_next_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, best_next_actions).squeeze(-1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        priorities = (Q_targets - Q_expected).abs().cpu().detach().numpy()
        self.memory.update_priorities(sample_indices, priorities)

    def train(self, n_episodes=250, max_t=1000, gamma=0.99, tau=0.005):
        env = gym.make('CarRacing-v2', continuous=False) 
        state, _ = env.reset() 
        frame_idx = 0
        for i_episode in range(1, n_episodes + 1):
            self.qnetwork_local.train()
            state, _ = env.reset()
            score = 0
            for t in range(max_t):
                frame_idx += 1
                self.beta = min(self.beta_end, self.beta_start + frame_idx * (self.beta_end - self.beta_start) / self.beta_frames) 
                action = self.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.memory.add(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if len(self.memory) > self.memory.batch_size and t%4==0:
                    experiences = self.memory.sample(self.beta)
                    self.learn(experiences, gamma, experiences[5])
                    self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)
                if  done:               
                    break
            self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon) 
            print(f"Episode {i_episode}\tScore: {(score):.2f}\tEpsilon: {self.epsilon:.2f}\tBeta: {self.beta:.2f}") 

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self):
        torch.save(self.qnetwork_local.state_dict(), 'model.pt')
        print("Model saved successfully.")

    def load(self):
        try:
            self.qnetwork_local.load_state_dict(torch.load('model.pt', map_location=self.device))
            self.epsilon = 0.00
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")
        
    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
