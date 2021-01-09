import sys
import time
import numpy as np
import gym
import torch
import torch.nn.functional as F
from gym import wrappers
from torch.utils.tensorboard import SummaryWriter
from collections import  deque
from utils import time_format
from torch.optim import Adam
from framestack import FrameStack
from models import QNetwork, SACActor, Encoder
from replay_buffer import ReplayBuffer


class SACAgent():
    def __init__(self, action_size, state_size, config):
        self.seed = config["seed"]
        torch.manual_seed(self.seed)
        np.random.seed(seed=self.seed)
        self.env = gym.make(config["env_name"])
        self.env = FrameStack(self.env, config)
        self.env.seed(self.seed)
        self.action_size = action_size
        self.state_size = state_size
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.history_length = config["history_length"]
        self.size = config["size"]
        if not torch.cuda.is_available():
            config["device"] == "cpu"
        self.device = config["device"]
        self.eval = config["eval"]
        self.vid_path = config["vid_path"]
        print("actions size ", action_size)
        self.critic = QNetwork(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.q_optim = torch.optim.Adam(self.critic.parameters(), config["lr_critic"])
        self.target_critic = QNetwork(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=config["lr_alpha"])
        self.policy = SACActor(state_size, action_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=config["lr_policy"])
        self.encoder = Encoder(config).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), self.lr)
        self.episodes = config["episodes"]
        self.memory = ReplayBuffer((self.history_length, self.size, self.size), (1, ), config["buffer_size"], config["image_pad"], self.seed, self.device)
        pathname = config["seed"]
        tensorboard_name = str(config["res_path"]) + '/runs/' + str(pathname)
        self.writer = SummaryWriter(tensorboard_name)
        self.steps= 0
        self.target_entropy = -torch.prod(torch.Tensor(action_size).to(self.device)).item()

    def act(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            state = state.type(torch.float32).div_(255)
            self.encoder.eval()
            state = self.encoder.create_vector(state)
            self.encoder.train()
            if evaluate is False:
                action = self.policy.sample(state)
            else:
                action_prob, _ = self.policy(state)
                action = torch.argmax(action_prob)
                action = action.cpu().numpy()
                return action
            # action = np.clip(action, self.min_action, self.max_action)
            action = action.cpu().numpy()[0]
        return action
    
    def train_agent(self):
        average_reward = 0
        scores_window = deque(maxlen=100)
        t0 = time.time()
        for i_epiosde in range(1, self.episodes):
            episode_reward = 0
            state = self.env.reset()
            t = 0
            while True:
                t += 1
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                if i_epiosde > 10:
                    self.learn()
                self.memory.add(state, reward, action, next_state, done)
                state = next_state
                if done:
                    scores_window.append(episode_reward)
                    break
            if i_epiosde % self.eval == 0:
                self.eval_policy()
            ave_reward = np.mean(scores_window)
            print("Epiosde {} Steps {} Reward {} Reward averge{:.2f} Time {}".format(i_epiosde, t, episode_reward, np.mean(scores_window), time_format(time.time() - t0)))
            self.writer.add_scalar('Aver_reward', ave_reward, self.steps)
            
    
    def learn(self):
        self.steps += 1
        states, rewards, actions, next_states, dones = self.memory.sample(self.batch_size)
        states = states.type(torch.float32).div_(255)
        states = self.encoder.create_vector(states)
        states_detached = states.detach()
        qf1, qf2 = self.critic(states)
        q_value1 = qf1.gather(1, actions)
        q_value2 = qf2.gather(1, actions)
        
        with torch.no_grad():
            next_states = next_states.type(torch.float32).div_(255)
            next_states = self.encoder.create_vector(next_states)
            q1_target, q2_target = self.target_critic(next_states)
            min_q_target = torch.min(q1_target, q2_target)
            next_action_prob, next_action_log_prob = self.policy(next_states)
            next_q_target = (next_action_prob * (min_q_target - self.alpha * next_action_log_prob)).sum(dim=1, keepdim=True)
            next_q_value = rewards + (1 - dones) * self.gamma * next_q_target


        # --------------------------update-q--------------------------------------------------------
        loss = F.mse_loss(q_value1, next_q_value) + F.mse_loss(q_value2, next_q_value) 
        self.q_optim.zero_grad() 
        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.q_optim.step()
        self.encoder_optimizer.zero_grad()
        self.writer.add_scalar('loss/q', loss, self.steps)


        # --------------------------update-policy--------------------------------------------------------
        action_prob, log_action_prob = self.policy(states_detached)
        with torch.no_grad():
            q_pi1, q_pi2 = self.critic(states_detached)
            min_q_values = torch.min(q_pi1, q_pi2)
        #policy_loss = (action_prob *  ((self.alpha * log_action_prob) - min_q_values).detach()).sum(dim=1).mean()
        policy_loss = (action_prob *  ((self.alpha * log_action_prob) - min_q_values)).sum(dim=1).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.writer.add_scalar('loss/policy', policy_loss, self.steps)

        # --------------------------update-alpha--------------------------------------------------------
        alpha_loss =(action_prob.detach() *  (-self.log_alpha * (log_action_prob + self.target_entropy).detach())).sum(dim=1).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.writer.add_scalar('loss/alpha', alpha_loss, self.steps)
        self.soft_udapte(self.critic, self.target_critic)
        self.alpha = self.log_alpha.exp()


    
    def soft_udapte(self, online, target):
        for param, target_parm in zip(online.parameters(), target.parameters()):
            target_parm.data.copy_(self.tau * param.data + (1 - self.tau) * target_parm.data)

    def eval_policy(self, eval_episodes=4):
        env = gym.make(self.env_name)
        env  = wrappers.Monitor(env, str(self.vid_path) + "/{}".format(self.steps), video_callable=lambda episode_id: True,force=True)
        average_reward = 0
        scores_window = deque(maxlen=100)
        for i_epiosde in range(eval_episodes):
            print("Eval Episode {} of {} ".format(i_epiosde, eval_episodes))
            episode_reward = 0
            state = env.reset()
            while True: 
                action = self.act(state, evaluate=True)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
            scores_window.append(episode_reward)
        average_reward = np.mean(scores_window)
        self.writer.add_scalar('Eval_reward', average_reward, self.steps)
