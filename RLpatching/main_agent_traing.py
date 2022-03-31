from ast import arg
from Re_buffer import *
import os
import critic_q
import argparse
from critic_q import Critic
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from numpy import *
from actor_pre_training.actor_pre_training import Agent_Supervised_Actor as Model_p
from critic_pre_training.critic_pre_training import Agent_Supervised_Critic as Model_r
from actor_pre_training.actor_pre_training import supervised_train_actor
from critic_pre_training.critic_pre_training import supervised_train_critic
from grid_agent import GridAgent
from cmath import inf
from utilize.form_action import *
from Environment.base_env import Environment
from utilize.settings import settings
import csv
import torch.utils.data as Data
import random
from dispatching_necessity_evaluation import *
import sys
sys.path.append("/root/lixinhang/Conformer")
import Conformer
from Conformer import Conformer


start_idx_set = 42312
select=True
R=0.9
L=0.9
N=40

def get_parsers():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
    #parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--test_iteration', default=10, type=int)

    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float) # discounted factor
    parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
    parser.add_argument('--batch_size', default=512, type=int) # mini batch size
    parser.add_argument('--seed', default=False, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)
    # optional parameters

    parser.add_argument('--sample_frequency', default=2000, type=int)
    #parser.add_argument('--render', default=False, type=bool) # show UI or not
    parser.add_argument('--log_interval', default=50, type=int) #

    #parser.add_argument('--load', default=False, type=bool) # load model
    parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
    #parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--max_episode', default=100000, type=int) # num of games
    parser.add_argument('--print_log', default=5, type=int)
    parser.add_argument('--update_iteration', default=100, type=int)
    parser.add_argument('--pre_training', action="store_true", default=False, help="pre training")
    parser.add_argument('--exp_name', type=str, default="0323_CR_R9_L9_N40")
    parser.add_argument('--format', type=int, default=0, help='the format of datas')
    return parser.parse_args()


def sep(_obs):
    _list = _obs.a_ex + _obs.p_ex + _obs.q_ex + _obs.v_ex + _obs.rho + _obs.grid_loss + _obs.nextstep_load_p + \
           _obs.load_q + _obs.nextstep_renewable_gen_p_max + _obs.gen_q
    return _list


def sep_all(_obs, repm, format):
    idx = start_idx_set + _obs.timestep
    row_y = []
    if format==0:
        next_ten_step_renewable_max=repm.pre(t=idx)
        for i in range(len(next_ten_step_renewable_max)):

            row_y += list(next_ten_step_renewable_max[i])
    elif format==1:
        for i in range(10):
            row_y += [0 for _ in range(18)]
    elif format==2:
        row_y += _obs.nextstep_renewable_gen_p_max
        listIndexs = list(range(idx+3, idx+11+1))
        with open('data/max_renewable_gen_p.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            reader_rows = [row for row in reader]
            for index in listIndexs:
                row_y_temp = []
                row = reader_rows[index]
                for num in row:
                    row_y_temp.append(float(num))
                row_y += row_y_temp
    elif format == 3:
        row_y = []
        next_ten_step_renewable_max=repm.pre(t=idx)
        row_y += list(next_ten_step_renewable_max[0])
        for i in range(9):
            row_y += [0 for _ in range(18)]

    _list = _obs.a_ex + _obs.p_ex + _obs.q_ex + _obs.v_ex + _obs.p_or + _obs.q_or + _obs.v_or + _obs.rho + \
           _obs.grid_loss + _obs.nextstep_load_p + _obs.load_q + row_y + _obs.gen_q

    return _list


def change_adj_p(obs, actions_p, num_gen):
        adjust_gen_p_action_space = obs.action_space['adjust_gen_p']
        min_adj_p = adjust_gen_p_action_space.low
        max_adj_p = adjust_gen_p_action_space.high
        for i in range(num_gen):
            if min_adj_p[i] == -inf:
                actions_p[i] = 0
                continue
            if min_adj_p[i] == inf:
                actions_p[i] = 0
                continue

            if actions_p[i] < min_adj_p[i]:
                actions_p[i] = min_adj_p[i]
            elif actions_p[i] > max_adj_p[i]:
                actions_p[i] = max_adj_p[i]

        return actions_p


def wrap_action(adjust_gen_p, adjust_gen_v):
        act = {
            'adjust_gen_p': adjust_gen_p,
            'adjust_gen_v': adjust_gen_v
        }
        return act


def change_adj_v(obs, actions_v, num_gen):
        adjust_gen_v_action_space = obs.action_space['adjust_gen_v']
        min_adj_v = adjust_gen_v_action_space.low
        max_adj_v = adjust_gen_v_action_space.high
        for i in range(num_gen):
            if min_adj_v[i] == -inf or min_adj_v[i] == inf:
                actions_v[i] = 0
                continue
            if actions_v[i] < min_adj_v[i]:
                actions_v[i] = min_adj_v[i]
            elif actions_v[i] > max_adj_v[i]:
                actions_v[i] = max_adj_v[i]
        return actions_v



def str_list_to_float_list(str_list):
    n = 0
    while n < len(str_list):
        str_list[n] = float(str_list[n])
        n += 1
    return (str_list)


class Agent():
    def __init__(self, pre_training=True):
   
        self.num_agent = 54
        self.rand=False
        self.device = torch.device('cuda')



        self.actor_p = Model_p().to(self.device)
        self.actor_p_target=Model_p().to(self.device)

        self.actor_optimizer = optim.Adam(self.actor_p.parameters(), lr=1e-7)
        self.StepLR_actor = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.8)


        self.critic_p = Model_r().to(self.device)
        self.critic_p_target = Model_r().to(self.device)

        self.critic_optimizer = optim.Adam(self.critic_p.parameters(), lr=1e-5)
        self.StepLR_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.8)



        self.replay_buffer = Replay_buffer(args)


        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        
        if pre_training:
            self.is_pre = self.pre_training()
        else:
            self.is_pre = False
     
    def initial_repm(self, start_index):
        self.repm=REPM(start_index=start_index)

    def pre_training(self):
        print("Pre training...")
        actor_result = supervised_train_actor(self.actor_p, format=args.format, exp_name=args.exp_name)
#        critic_result = supervised_train_critic(self.critic_p, format=args.format, exp_name=args.exp_name)
        return actor_result

    def load(self):
        if self.is_pre:
            print("Load from pre training model...")
            self.actor_p_target.load_state_dict(self.actor_p.state_dict())
            self.critic_p_target.load_state_dict(self.critic_p.state_dict())
            self.is_pre = False
        else:
            if not os.path.exists(os.path.join('best_model', args.exp_name, 'actor_best_10.pth')):
                print("Load the actor from the history model...")
                model_path_p = os.path.join('best_model', args.exp_name, "train_model_p_all_10.pth")

                self.actor_p.load_state_dict(torch.load(model_path_p,  map_location='cuda'))
                self.actor_p_target.load_state_dict(self.actor_p.state_dict())
            else:
                print("Load the actor from best model...")
                # model_path_p = os.path.join('best_model_10/actor_best_10.pth')
                model_path_p = os.path.join('best_model', args.exp_name, 'actor_best_10.pth')
                self.actor_p.load_state_dict(torch.load(model_path_p,  map_location='cuda'))
                self.actor_p_target.load_state_dict(self.actor_p.state_dict())

            if not os.path.exists(os.path.join('best_model', args.exp_name, 'critic_best_10.pth')):
                print("Load the critic from the history model...")
                #self.critic_p_target.load_state_dict(self.critic_p.state_dict())
                model_path_r = os.path.join('best_model', args.exp_name, "train_model_r_all_10.pth")
               # model_path_r = os.path.join('best_model_10/critic_best_10.pth')
                self.critic_p.load_state_dict(torch.load(model_path_r,  map_location='cuda'))
                self.critic_p_target.load_state_dict(self.critic_p.state_dict())
            else:
                print("Load the critic from best model...")
                # self.actor_optimizer = optim.Adam(self.actor_p.parameters(), lr=1e-6)
                # StepLR_ = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.45)
                # model_path_r = os.path.join('best_model_10/critic_best_10.pth')
                model_path_r = os.path.join('best_model', args.exp_name, 'critic_best_10.pth')
                self.critic_p.load_state_dict(torch.load(model_path_r,  map_location='cuda'))
                self.critic_p_target.load_state_dict(self.critic_p.state_dict())
                # self.critic_optimizer = optim.Adam(self.critic_p.parameters(), lr=1e-6)
                # StepLR_ = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.45)
            self.actor_p = self.actor_p.to(self.device)
            self.actor_p_target = self.actor_p_target.to(self.device)
            self.critic_p = self.critic_p.to(self.device)
            self.critic_p_target = self.critic_p_target.to(self.device)

    def act(self, obs, state_all_repm=None):
        action_p, action_v = self.test(obs, state_all_repm)
        ret_action = wrap_action(action_p, action_v)
        return ret_action

    def test(self, obs, state_all_repm=None):


        state_all = state_all_repm

        self.actor_p.eval()


        state_all = state_all.to(self.device)
        output_p = self.actor_p(state_all)

        output_p = output_p.cpu().detach().numpy().tolist()


        output_p = output_p[0]


        action_p_orig = []
        action_v_orig=[]


        for i in range(self.num_agent):
            action_p_orig.append(output_p[i] - obs.gen_p[i])
            # action_v_orig.append(output_v[i] - obs.gen_v[i])
            action_v_orig.append(0)


        action_v = change_adj_v(obs, action_v_orig, self.num_agent)

        

        if self.rand:
            # random_list = random.sample(range(0,self.num_agent),15)
            for i in range(54):
                action_p_orig[i] = action_p_orig[i] + random.uniform(-3,3)
#	            action_p_orig[random_list[i]] = -1 * action_p_orig[random_list[i]]
#	            action_p_orig[i] = action_p_orig[i] + random.uniform(-5,5)

        if select:
                print("selecting...")
                action_p_orig=dispatching_necessity_evaluation(obs,output_p,action_p_orig,fiR=R,fiL=L,ndis=N)
        action_p = change_adj_p(obs,action_p_orig,self.num_agent)


        return action_p, action_v

    def train(self,time_a,time_c):
        all_actor_loss=0
        all_critic_loss = 0

        self.actor_p_target.eval()
        self.critic_p_target.eval()

        # Sample replay buffer
        x, y, u, r, d = self.replay_buffer.sample(args.batch_size)

        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(1-d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)


        # Compute the target Q value

        target_Q = self.critic_p_target(next_state, self.actor_p_target(next_state))
        target_Q = reward + (done * args.gamma * target_Q).detach()



        current_Q = self.critic_p(state, action)

        # Compute critic loss
        
        critic_loss = F.mse_loss(current_Q, target_Q)
        all_critic_loss =all_critic_loss+critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        print("current_lr: %s" % (self.critic_optimizer.state_dict()['param_groups'][0]['lr']))


        # Compute actor loss
        actor_loss = -self.critic_p(state, self.actor_p(state)).mean()
        all_actor_loss = all_actor_loss + actor_loss.item()
        


        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        

        # Update the frozen target models

        if time_c % 5 ==0:
            for param, target_param in zip(self.critic_p.parameters(), self.critic_p_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        if time_a % 2 ==0:
            for param, target_param in zip(self.actor_p.parameters(), self.actor_p_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1

        return all_critic_loss, all_actor_loss

    def save_a(self):
        print("Save the actor network...")
        save_path = os.path.join('best_model_10/', args.exp_name)
        torch.save(self.actor_p.state_dict(), save_path+'/actor_best_10.pth')
        self.StepLR_actor.step()

    def save_c(self):
        print("Save the critic network...")
        # save_path='best_model_10/'
        save_path = os.path.join('best_model_10/', args.exp_name)
        torch.save(self.critic_p.state_dict(), save_path+'/critic_best_10.pth')
        self.StepLR_critic.step()

    def test_result(self):
        self.actor_p.eval()
        total_reward = 0
        step = 0
        env = Environment(settings, "EPRIReward")
        obs, start_idx = env.reset(start_sample_idx=42312)
        self.initial_repm(start_index=42312)
        
        state_p = torch.Tensor([sep_all(obs, agent.repm, args.format)])
        test_total_reward_t_lists = {"EPRIReward":[], "line_over_flow_reward":[], "renewable_consumption_reward":[] ,
                            "running_cost_reward":[] ,"balanced_gen_reward":[] ,"gen_reactive_power_reward":[] ,"sub_voltage_reward":[],"running_cost":[]}
        while True:
            action = self.act(obs, state_p)
            adj_action_p=action['adjust_gen_p']
            action_p=[]
            for gen_i in range(len(adj_action_p)):
                action_p.append(adj_action_p[gen_i]+obs.gen_p[gen_i])
            print("test****************",step)
            #print(action_p)
            next_obs, reward_lists, done, info = env.step(action)
            # print("List:")
            # print(reward_lists)
            if reward_lists == 0:
                reward = 0
            else:
                reward = reward_lists["EPRIReward"]
            total_reward += reward
            if reward != 0:
                test_total_reward_t_lists["EPRIReward"].append(reward_lists["EPRIReward"])
                test_total_reward_t_lists["line_over_flow_reward"].append(reward_lists["line_over_flow_reward"])
                test_total_reward_t_lists["renewable_consumption_reward"].append(reward_lists["renewable_consumption_reward"])
                test_total_reward_t_lists["running_cost_reward"].append(reward_lists["running_cost_reward"])
                test_total_reward_t_lists["balanced_gen_reward"].append(reward_lists["balanced_gen_reward"])
                test_total_reward_t_lists["gen_reactive_power_reward"].append(reward_lists["gen_reactive_power_reward"])
                test_total_reward_t_lists["sub_voltage_reward"].append(reward_lists["sub_voltage_reward"])
                test_total_reward_t_lists["running_cost"] .append( reward_lists["running_cost"])
            print("step_reward:",reward)
#   r_pre=total_reward
            if abs(reward-0)<0.00001:
                reward=-2
        
            state_p_next = torch.Tensor([sep_all(next_obs, agent.repm, args.format)])
            #if args.render and i >= args.render_interval: env.render()
            if reward>0.5 or len(self.replay_buffer.storage) < 500:
                self.replay_buffer.push((state_p, state_p_next, action_p, reward, np.float(done)))
            state_p = state_p_next

            obs = next_obs
            if done:
                break
            step += 1
        return total_reward, step, test_total_reward_t_lists


if __name__ == '__main__':
    args = get_parsers()
    save_Test=False
    if not os.path.exists(os.path.join('best_model/', args.exp_name)):
        os.makedirs(os.path.join('best_model/', args.exp_name))
        print("make the new dirs...")
    pre_tra = False
    agent=Agent(pre_training=pre_tra)
    total_step = 0
    r_best = 0
    c_loss_best = 100
    time_a = 0
    time_c = 0
    if pre_tra:
        with open(os.path.join('best_model', args.exp_name, "result_online_best_reward.csv"),"w",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode","train_times","update_times","steps","EPRIReward","line_over_flow_reward","renewable_consumption_rewards","running_cost_reward","balanced_gen_reward","gen_reactive_power_reward","sub_voltage_reward","total_cost","A loss","C loss"])

        with open(os.path.join('best_model', args.exp_name, "result_online_loss.csv"),"w",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode","train_times","update_times","steps","total reward","A loss","C loss"])

        with open(os.path.join('best_model', args.exp_name, "result_best_epoch_reward.csv"),"w",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["step","EPRIReward","line_over_flow_reward","renewable_consumption_rewards","running_cost_reward","balanced_gen_reward","gen_reactive_power_reward","sub_voltage_reward","cost"])

        with open(os.path.join('best_model', args.exp_name, "test_reward.csv"),"w",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode","update_times","steps","EPRIReward","line_over_flow_reward","renewable_consumption_rewards","running_cost_reward","balanced_gen_reward","gen_reactive_power_reward","sub_voltage_reward","cost"])

        with open(os.path.join('best_model', args.exp_name, "result_loss.csv"),"w",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode","Average a loss","Average c loss"])
        # with open(os.path.join('best_model_10', args.exp_name, "result_episode_reward.csv"),"w",newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(["episode","EPRIReward","line_over_flow_reward","renewable_consumption_rewards","running_cost_reward","balanced_gen_reward","gen_reactive_power_reward","sub_voltage_reward"])
    for i in range(args.max_episode):
        total_reward = 0
        step = 0
        env = Environment(settings, "EPRIReward")
        obs, start_idx = env.reset(start_sample_idx=42312)
        agent.initial_repm(start_index=42312)
        agent.load()
        state_p = torch.Tensor([sep_all(obs, agent.repm, args.format)])
        test_total_reward_t_lists = {"EPRIReward":[], "line_over_flow_reward":[], "renewable_consumption_reward":[] ,
                            "running_cost_reward":[] ,"balanced_gen_reward":[] ,"gen_reactive_power_reward":[] ,"sub_voltage_reward":[],"running_cost":[]}
        while True:
            action = agent.act(obs, state_p)
            adj_action_p=action['adjust_gen_p']
            action_p=[]
            for gen_i in range(len(adj_action_p)):
                action_p.append(adj_action_p[gen_i]+obs.gen_p[gen_i])
            print("****************",step)
            next_obs, reward_lists, done, info = env.step(action)
            if reward_lists == 0:
                reward = 0
            else:
                reward = reward_lists["EPRIReward"]
            total_reward += reward
            if reward != 0:
                test_total_reward_t_lists["EPRIReward"].append(reward_lists["EPRIReward"])
                test_total_reward_t_lists["line_over_flow_reward"] .append( reward_lists["line_over_flow_reward"])
                test_total_reward_t_lists["renewable_consumption_reward"] .append( reward_lists["renewable_consumption_reward"])
                test_total_reward_t_lists["running_cost_reward"] .append( reward_lists["running_cost_reward"])
                test_total_reward_t_lists["balanced_gen_reward"] .append( reward_lists["balanced_gen_reward"])
                test_total_reward_t_lists["gen_reactive_power_reward"] .append( reward_lists["gen_reactive_power_reward"])
                test_total_reward_t_lists["sub_voltage_reward"] .append( reward_lists["sub_voltage_reward"])
                test_total_reward_t_lists["running_cost"] .append( reward_lists["running_cost"])
                
            print("step_reward:",reward)

            if abs(reward-0)<0.00001:
               reward=-2
        
            state_p_next = torch.Tensor([sep_all(next_obs, agent.repm, args.format)])

            if reward>-5 or len(agent.replay_buffer.storage) < 1000:
                agent.replay_buffer.push((state_p, state_p_next, action_p, reward, np.float(done)))
            state_p = state_p_next
            obs = next_obs
            if done:
                break
            step += 1

        print("total_reward:",total_reward)
        total_step += step + 1
        if r_best>=50:
            agent.rand=True
        if save_Test or total_reward>r_best:
            with open(os.path.join('best_model', args.exp_name, "test_reward.csv"),"a+",newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([ i,time_a,step,
                                          sum(test_total_reward_t_lists["EPRIReward"]),
                                          sum(test_total_reward_t_lists["line_over_flow_reward"]),
                                          sum(test_total_reward_t_lists["renewable_consumption_reward"]),
                                          sum(test_total_reward_t_lists["running_cost_reward"]),
                                          sum(test_total_reward_t_lists["balanced_gen_reward"]),
                                          sum(test_total_reward_t_lists["gen_reactive_power_reward"]),
                                          sum(test_total_reward_t_lists["sub_voltage_reward"]),
                                          sum(test_total_reward_t_lists["running_cost"])])
            save_Test=False
        if total_reward>r_best:
            r_best=total_reward
            with open(os.path.join('best_model', args.exp_name, "result_online_best_reward.csv"),"a+",newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([ i, 0, time_a,step,
                                          sum(test_total_reward_t_lists["EPRIReward"]),
                                          sum(test_total_reward_t_lists["line_over_flow_reward"]),
                                          sum(test_total_reward_t_lists["renewable_consumption_reward"]),
                                          sum(test_total_reward_t_lists["running_cost_reward"]),
                                          sum(test_total_reward_t_lists["balanced_gen_reward"]),
                                          sum(test_total_reward_t_lists["gen_reactive_power_reward"]),
                                          sum(test_total_reward_t_lists["sub_voltage_reward"]),
                                          sum(test_total_reward_t_lists["running_cost"]),
                                          0, 0])
            with open(os.path.join('best_model', args.exp_name, "result_best_epoch_reward.csv"), "w",
                      newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Step", "EPRIReward", "line_over_flow_reward", "renewable_consumption_rewards",
                                 "running_cost_reward", "balanced_gen_reward", "gen_reactive_power_reward",
                                 "sub_voltage_reward","cost"])
                for step_num in range(len(test_total_reward_t_lists["EPRIReward"])):
                    writer.writerow([step_num, test_total_reward_t_lists["EPRIReward"][step_num],
                                     test_total_reward_t_lists["line_over_flow_reward"][step_num],
                                     test_total_reward_t_lists["renewable_consumption_reward"][step_num],
                                     test_total_reward_t_lists["running_cost_reward"][step_num],
                                     test_total_reward_t_lists["balanced_gen_reward"][step_num],
                                     test_total_reward_t_lists["gen_reactive_power_reward"][step_num],
                                     test_total_reward_t_lists["sub_voltage_reward"][step_num],
                                     test_total_reward_t_lists["running_cost"][step_num]])

        if len(agent.replay_buffer.storage)> args.batch_size:
            total_c_loss = 0
            total_a_loss = 0
            iterations = 0
            for train_times in range(args.update_iteration):
                print("****************",train_times,"****************")
                c_loss,a_loss=agent.train(time_a,time_c)
                total_c_loss += c_loss
                total_a_loss += a_loss
                iterations += 1

                total_reward_test, step_test, reward_lists = agent.test_result()

                if total_reward_test > r_best:
                    agent.save_a()
                    r_pre = total_reward_test
                    time_a = time_a + 1
                    save_Test = True
                    with open(os.path.join('best_model', args.exp_name, "result_online_best_reward.csv"),"a+",newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([i, train_times, time_a, step_test,
                                         sum(reward_lists["EPRIReward"]),
                                         sum(reward_lists["line_over_flow_reward"]),
                                         sum(reward_lists["renewable_consumption_reward"]),
                                         sum(reward_lists["running_cost_reward"]),
                                         sum(reward_lists["balanced_gen_reward"]),
                                         sum(reward_lists["gen_reactive_power_reward"]),
                                         sum(reward_lists["sub_voltage_reward"]),
                                         sum(reward_lists["running_cost"]),
                                         a_loss, c_loss])

                    if total_reward_test > 0:
                        with open(os.path.join('best_model', args.exp_name, "result_best_epoch_reward.csv"), "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["Step", "EPRIReward","line_over_flow_reward","renewable_consumption_rewards","running_cost_reward","balanced_gen_reward","gen_reactive_power_reward","sub_voltage_reward","cost"])
                            for step_num in range(len(reward_lists["EPRIReward"])):
                                writer.writerow([step_num,
                                                 reward_lists["EPRIReward"][step_num],
                                                 reward_lists["line_over_flow_reward"][step_num],
                                                 reward_lists["renewable_consumption_reward"][step_num],
                                                 reward_lists["running_cost_reward"][step_num],
                                                 reward_lists["balanced_gen_reward"][step_num],
                                                 reward_lists["gen_reactive_power_reward"][step_num],
                                                 reward_lists["sub_voltage_reward"][step_num],
                                                 reward_lists["running_cost"][step_num]])
                
                if c_loss < c_loss_best:
                    agent.save_c()
                    c_loss_best = c_loss
                    time_c = time_c + 1
                    with open(os.path.join('best_model', args.exp_name, "result_online_loss.csv"), "a+", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([i, train_times, time_c,step_test, total_reward_test, a_loss, c_loss])
                

                print("EXP_NAME: {} Total T:{} Episode: \t{} steps: \t{} Total Reward: \t{:0.2f} A_loss: \t{:0.2f} C_loss: \t{:0.2f} R_PRE: \t{:0.2f} Test_reward: \t{:0.2f}".format(args.exp_name, total_step, i, step_test,
                                                                                            total_reward_test,a_loss,c_loss,r_pre,total_reward_test))

                if step_test<50 and r_pre > 200:
                    break


            with open(os.path.join('best_model', args.exp_name, "result_loss.csv"), "a+", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([i, total_a_loss/iterations, total_c_loss/iterations])
        else:
            print("EXP_NAME: {} Total T:{} Episode: \t{} steps: \t{} Total Reward: \t{:0.2f} R_PRE: \t{:0.2f}".format(args.exp_name, total_step, i, step,
                                                                                        total_reward,r_pre))




