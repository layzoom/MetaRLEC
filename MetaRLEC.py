# coding=utf-8

import os
import logging
import platform
import random
from tqdm import tqdm
import numpy as np
import torch

from common import BaseLearner, Tensor, consts
from frame import Actor, EpisodicCritic, Reward, DenseCritic, Meta_Critic, Hot_Plug
from frame import score_function as Score_Func
from utils.data_loader import DataGenerator
from utils.graph_analysis import get_graph_from_order, pruning_by_coef
from utils.graph_analysis import pruning_by_coef_2nd
from common.validator import check_args_value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


class MetaRLEC(BaseLearner):

    @check_args_value(consts.MetaRLEC_VALID_PARAMS)
    def __init__(self, batch_size=64, input_dim=100, embed_dim=256,
                 normalize=False,
                 encoder_name='transformer',
                 encoder_heads=8,
                 encoder_blocks=3,
                 encoder_dropout_rate=0.1,
                 decoder_name='lstm',
                 reward_mode='episodic',
                 reward_score_type='BIC',
                 reward_regression_type='LR',
                 reward_gpr_alpha=1.0,
                 iteration=500,
                 lambda_iter_num=50,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 alpha=0.99,  # for score function
                 init_baseline=-1.0,
                 random_seed=0,
                 device_type='gpu',
                 device_ids=0
                 ):
        super(MetaRLEC, self).__init__()
        self.batch_size             = batch_size
        self.input_dim              = input_dim
        self.embed_dim              = embed_dim
        self.normalize              = normalize
        self.encoder_name           = encoder_name
        self.encoder_heads          = encoder_heads
        self.encoder_blocks         = encoder_blocks
        self.encoder_dropout_rate   = encoder_dropout_rate
        self.decoder_name           = decoder_name
        self.reward_mode            = reward_mode
        self.reward_score_type      = reward_score_type
        self.reward_regression_type = reward_regression_type
        self.reward_gpr_alpha       = reward_gpr_alpha
        self.iteration              = iteration
        self.lambda_iter_num        = lambda_iter_num
        self.actor_lr               = actor_lr
        self.critic_lr              = critic_lr
        self.alpha                  = alpha
        self.init_baseline          = init_baseline
        self.random_seed            = random_seed
        self.device_type            = device_type
        self.device_ids             = device_ids
        if reward_mode == 'dense':
            self.avg_baseline = torch.tensor(init_baseline, requires_grad=False)

        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device


    def learn(self, data, columns=None, **kwargs) -> None:
        X = Tensor(data, columns=columns)
        self.n_samples = X.shape[0]
        self.seq_length = X.shape[1] # seq_length == n_nodes
        if X.shape[1] > self.batch_size:
            raise ValueError(f'The `batch_size` must greater than or equal to '
                             f'`n_nodes`, but got '
                             f'batch_size: {self.batch_size}, '
                             f'n_nodes: {self.seq_length}.')
        self.dag_mask = getattr(kwargs, 'dag_mask', None)
        causal_matrix = self._rl_search(X)
        self.causal_matrix = Tensor(causal_matrix,
                                    index=X.columns,
                                    columns=X.columns)

    def _rl_search(self, X) -> torch.Tensor:

        set_seed(self.random_seed)
        logging.info('Python version is {}'.format(platform.python_version()))

        # generate observed data
        data_generator = DataGenerator(dataset=X,
                                       normalize=self.normalize,
                                       device=self.device)
        # Instantiating an Actor
        actor = Actor(input_dim=self.input_dim,
                      embed_dim=self.embed_dim,
                      encoder_blocks=self.encoder_blocks,
                      encoder_heads=self.encoder_heads,
                      encoder_name=self.encoder_name,
                      decoder_name=self.decoder_name,
                      max_length=self.seq_length,
                      device=self.device)
        self.hotplug = Hot_Plug(actor.encoder)
        # Instantiating an Critic
        if self.reward_mode == 'episodic':
            critic = EpisodicCritic(input_dim=self.embed_dim,
                                    device=self.device)
        else:
            critic = DenseCritic(input_dim=self.embed_dim,
                                 output_dim=self.embed_dim,
                                 device=self.device)

        meta_critic = Meta_Critic(state_dim=self.embed_dim,device=self.device)

        # Instantiating an Reward
        reward =Reward(input_data=data_generator.dataset.cpu().detach().numpy(),
                       reward_mode=self.reward_mode,
                       score_type=self.reward_score_type,
                       regression_type=self.reward_regression_type,
                       alpha=self.reward_gpr_alpha)
        # Instantiating an Optimizer

        optimizer_critic = torch.optim.Adam(params=critic.parameters(), lr=self.critic_lr)
        optimizer_meta_critic = torch.optim.Adam(params=meta_critic.parameters(), lr=self.critic_lr)
        # optimizer_actor = torch.optim.Adam(params=actor,lr=self.actor_lr)
        optimizer = torch.optim.Adam([
            {
                'params': actor.encoder.parameters(), 'lr': self.actor_lr
            },
            {
                'params': actor.decoder.parameters(), 'lr': self.actor_lr
            }
        ])

        max_reward = float('-inf')

        # logging.info(f'Shape of input batch: {self.batch_size}, '
        #              f'{self.seq_length}, {self.input_dim}')
        # logging.info(f'Shape of input batch: {self.batch_size}, '
        #              f'{self.seq_length}, {self.embed_dim}')
        # logging.info('Starting training.')
        print(f'Shape of input batch: {self.batch_size}, 'f'{self.seq_length}, {self.input_dim}')
        print(f'Shape of input batch: {self.batch_size}, 'f'{self.seq_length}, {self.embed_dim}')
        print('Starting training.')

        graph_batch_pruned = Tensor(np.ones((self.seq_length,self.seq_length)) -np.eye(self.seq_length))

        for i in tqdm(range(1, self.iteration + 1)):
            input_batch = data_generator.draw_batch(batch_size=self.batch_size,
                                                    dimension=self.input_dim).float()
            encoder_output = actor.encode(input=input_batch)
            decoder_output = actor.decode(input=encoder_output)
            actions, mask_scores, s_list, h_list, c_list = decoder_output
            batch_graphs = []
            action_mask_s = []
            for m in range(actions.shape[0]):
                zero_matrix = get_graph_from_order(actions[m].cpu())
                action_mask = np.zeros(zero_matrix.shape[0])
                for act in actions[m]:
                    action_mask_s.append(action_mask.copy())
                    action_mask += np.eye(zero_matrix.shape[0])[act]
                batch_graphs.append(zero_matrix)
            batch_graphs = np.stack(batch_graphs)
            action_mask_s = np.stack(action_mask_s)

            # Reward
            reward_output = reward.cal_rewards(batch_graphs, actions.cpu())
            reward_list, normal_batch_reward, max_reward_batch, td_target = reward_output

            if max_reward < max_reward_batch:
                max_reward = max_reward_batch

            # Critic
            prev_input = s_list.reshape((-1, self.embed_dim))
            prev_state_0 = h_list.reshape((-1, self.embed_dim))
            prev_state_1 = c_list.reshape((-1, self.embed_dim))

            action_mask_ =  action_mask_s.reshape((-1, self.seq_length))
            log_softmax = actor.decoder.log_softmax(input=prev_input,
                                                    position=actions,
                                                    mask=action_mask_,
                                                    state_0=prev_state_0,
                                                    state_1=prev_state_1)
            log_softmax = log_softmax.reshape((self.batch_size,self.seq_length)).T

            if self.reward_mode == 'episodic':
                critic.predict_env(stats_x=s_list[:, :-1, :])
                critic.predict_tgt(stats_y=s_list[:, 1:, :])
                critic.soft_replacement()
                td_target = td_target[::-1][:-1]

                actor_loss = Score_Func.episodic_actor_loss(
                    td_target=torch.tensor(td_target),
                    prediction_env=critic.prediction_env,
                    log_softmax=log_softmax,
                    device=self.device
                )
                critic_loss = Score_Func.episodic_critic_loss(
                    td_target=torch.tensor(td_target),
                    prediction_env=critic.prediction_env,
                    device=self.device
                )

            elif self.reward_mode == 'dense':
                log_softmax = torch.sum(log_softmax, 0)
                reward_mean = np.mean(normal_batch_reward)
                self.avg_baseline = self.alpha * self.avg_baseline + \
                                    (1.0 - self.alpha) * reward_mean
                predict_reward = critic.predict_reward(encoder_output=encoder_output)

                actor_loss = Score_Func.dense_actor_loss(normal_batch_reward,
                                                         self.avg_baseline,
                                                         predict_reward,
                                                         log_softmax,
                                                         device=self.device)
                critic_loss = Score_Func.dense_critic_loss(normal_batch_reward,
                                                           self.avg_baseline,
                                                           predict_reward,
                                                           device=self.device)
            else:
                raise ValueError(f"reward_mode must be one of ['episodic', "f"'dense'], but got {self.reward_mode}.")\

            # optimizer.zero_grad()
            # actor_loss.backward()
            # critic_loss.backward()
            # optimizer.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Part1 of Meta-test stage
            optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.hotplug.update(self.actor_lr)
            val_batch = data_generator.draw_batch(batch_size=self.batch_size, dimension=self.input_dim).float()
            encoder_output_val1 = actor.encode(input=val_batch)
            decoder_output_val1 = actor.decode(input=encoder_output_val1)
            actions_val1, mask_scores_val1, s_list_val1, h_list_val1, c_list_val1 = decoder_output_val1
            batch_graphs_val1 = []
            action_mask_s_val1 = []
            for m in range(actions_val1.shape[0]):
                zero_matrix_val1 = get_graph_from_order(actions_val1[m].cpu())
                action_mask_val1 = np.zeros(zero_matrix_val1.shape[0])
                for act in actions_val1[m]:
                    action_mask_s_val1.append(action_mask_val1.copy())
                    action_mask_val1 += np.eye(zero_matrix_val1.shape[0])[act]
                batch_graphs_val1.append(zero_matrix_val1)
            batch_graphs_val1 = np.stack(batch_graphs_val1)
            action_mask_s_val1 = np.stack(action_mask_s_val1)
            reward_output_val1 = reward.cal_rewards(batch_graphs_val1, actions_val1.cpu())
            reward_list_val1, normal_batch_reward_val1, max_reward_batch_val1, td_target_val1 = reward_output_val1

            prev_input_val1 = s_list_val1.reshape((-1, self.embed_dim))
            prev_state_0_val1 = h_list_val1.reshape((-1, self.embed_dim))
            prev_state_1_val1 = c_list_val1.reshape((-1, self.embed_dim))
            action_mask__val1 =  action_mask_s_val1.reshape((-1, self.seq_length))
            log_softmax_val1 = actor.decoder.log_softmax(input=prev_input_val1,
                                                    position=actions_val1,
                                                    mask=action_mask__val1,
                                                    state_0=prev_state_0_val1,
                                                    state_1=prev_state_1_val1)
            log_softmax_val1 = log_softmax_val1.reshape((self.batch_size,self.seq_length)).T
            actor_loss_val1 = 0
            actor_loss_val2 = 0
            if self.reward_mode == 'episodic':
                critic.predict_env(stats_x=s_list_val1[:, :-1, :])
                critic.predict_tgt(stats_y=s_list_val1[:, 1:, :])
                critic.soft_replacement()
                td_target_val1 = td_target_val1[::-1][:-1]

                actor_loss_val1 = Score_Func.episodic_actor_loss(
                    td_target=torch.tensor(td_target_val1),
                    prediction_env=critic.prediction_env,
                    log_softmax=log_softmax_val1,
                    device=self.device
                )
                critic_loss_val1 = Score_Func.episodic_critic_loss(
                    td_target=torch.tensor(td_target_val1),
                    prediction_env=critic.prediction_env,
                    device=self.device
                )
            print("actor_loss_val1")

            loss_auxiliary = meta_critic(encoder_output_val1)
            loss_auxiliary.backward(create_graph=True)
            self.hotplug.update(self.actor_lr)

            encoder_output_val2 = actor.encode(input=val_batch)
            decoder_output_val2 = actor.decode(input=encoder_output_val2)
            actions_val2, mask_scores_val2, s_list_val2, h_list_val2, c_list_val2 = decoder_output_val2
            batch_graphs_val2 = []
            action_mask_s_val2 = []
            for m in range(actions_val2.shape[0]):
                zero_matrix_val2 = get_graph_from_order(actions_val2[m].cpu())
                action_mask_val2 = np.zeros(zero_matrix_val2.shape[0])
                for act in actions_val2[m]:
                    action_mask_s_val2.append(action_mask_val2.copy())
                    action_mask_val2 += np.eye(zero_matrix_val2.shape[0])[act]
                batch_graphs_val2.append(zero_matrix_val2)
            batch_graphs_val2 = np.stack(batch_graphs_val2)
            action_mask_s_val2 = np.stack(action_mask_s_val2)
            reward_output_val2 = reward.cal_rewards(batch_graphs_val2, actions_val2.cpu())
            reward_list_val2, normal_batch_reward_val2, max_reward_batch_val2, td_target_val2 = reward_output_val2

            prev_input_val2 = s_list_val2.reshape((-1, self.embed_dim))
            prev_state_0_val2 = h_list_val2.reshape((-1, self.embed_dim))
            prev_state_1_val2 = c_list_val2.reshape((-1, self.embed_dim))
            action_mask__val2 =  action_mask_s_val2.reshape((-1, self.seq_length))
            log_softmax_val2 = actor.decoder.log_softmax(input=prev_input_val2,
                                                    position=actions_val2,
                                                    mask=action_mask__val2,
                                                    state_0=prev_state_0_val2,
                                                    state_1=prev_state_1_val2)
            log_softmax_val2 = log_softmax_val2.reshape((self.batch_size,self.seq_length)).T
            if self.reward_mode == 'episodic':
                critic.predict_env(stats_x=s_list_val2[:, :-1, :])
                critic.predict_tgt(stats_y=s_list_val2[:, 1:, :])
                critic.soft_replacement()
                td_target_val2 = td_target_val2[::-1][:-1]

                actor_loss_val2 = Score_Func.episodic_actor_loss(
                    td_target=torch.tensor(td_target_val2),
                    prediction_env=critic.prediction_env,
                    log_softmax=log_softmax_val2,
                    device=self.device
                )
                critic_loss_val2 = Score_Func.episodic_critic_loss(
                    td_target=torch.tensor(td_target_val2),
                    prediction_env=critic.prediction_env,
                    device=self.device
                )
            print("actor_loss_val2")

            utility = actor_loss_val1 - actor_loss_val2
            utility = torch.tanh(utility)
            loss_meta = -utility

            # Meta optimization of auxilary network
            optimizer_meta_critic.zero_grad()
            # grad_omega = torch.autograd.grad(loss_meta, meta_critic.parameters(), allow_unused=True)
            # for gradient, variable in zip(grad_omega, meta_critic.parameters()):
            #     variable.grad.data = gradient
            loss_meta.backward()
            optimizer_meta_critic.step()

            # update backward()
            # actor_optimizer.step()
            # actor_feature_optimizer.step()
            optimizer.step()
            self.hotplug.restore()

            # logging
            if i == 1 or i % consts.LOG_FREQUENCY == 0:
                # logging.info('[iter {}] max_reward: {:.4}, ''max_reward_batch: {:.4}'.format(i, max_reward, max_reward_batch))
                print('[iter {}] max_reward: {:.4}, ''max_reward_batch: {:.4}'.format(i, max_reward, max_reward_batch))
            if i == 1 or i % self.lambda_iter_num == 0:
                ls_kv = reward.update_all_scores()
                score_min, graph_int_key = ls_kv[0][1][0], ls_kv[0][0]
                print('[iter {}] score_min {:.4}'.format(i, score_min * 1.0))
                # logging.info('[iter {}] score_min {:.4}'.format(i, score_min * 1.0))
                graph_batch = get_graph_from_order(graph_int_key,
                                                   dag_mask=self.dag_mask)

                if self.reward_regression_type == 'LR':
                    graph_batch_pruned = pruning_by_coef(
                        graph_batch, data_generator.dataset.cpu().detach().numpy()
                    )
                elif self.reward_regression_type == 'QR':
                    graph_batch_pruned = pruning_by_coef_2nd(
                        graph_batch, data_generator.dataset.cpu().detach().numpy()
                    )
                else:
                    raise ValueError(f"reward_regression_type must be one of "
                                     f"['LR', 'QR'], but got "
                                     f"{self.reward_regression_type}.")

        return graph_batch_pruned

if __name__ == '__main__':
    # from datasets import load_dataset
    import scipy.io as scio
    from common import GraphDAG
    from metrics import MetricsDAG
    # X, true_dag, _ = load_dataset('IID_Test')
    # n = MetaRLEC()
    # n.learn(X)
    # GraphDAG(n.causal_matrix, true_dag)
    # met = MetricsDAG(n.causal_matrix, true_dag)
    # print(met.metrics)
    # print('----------------------------------------------sim15结束-------------------------------------------------')
    # true_dag = np.loadtxt('data/sub1_target.csv', delimiter=',')
    # data = scio.loadmat('data/sim1.mat')
    # dataset = np.array(data['ts'])
    # n = MetaRLEC()
    # n.learn(dataset)
    # GraphDAG(n.causal_matrix, true_dag)
    # met = MetricsDAG(n.causal_matrix, true_dag)
    # print(met.metrics)
    # print('----------------------------------------------sim1结束-------------------------------------------------')
    # true_dag = np.loadtxt('data/sim3_target.csv', delimiter=',')
    # # data = scio.loadmat('data/sim3.mat')
    # data = scio.loadmat("data/sim3.mat")
    # dataset = np.array(data['ts'])
    # for i in range(0,50):
    #     # Meta_X1 = M_dataset[(0 + i * 200): (600 + i * 200)]
    #     # X = dataset[0:200, :]
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     # n.meta_learn(X,Meta_X1)
    #     n.learn(X)
    #     GraphDAG(n.causal_matrix, true_dag)
    #     met = MetricsDAG(n.causal_matrix, true_dag)
    #     print(met.metrics)
    # print('----------------------------------------------sim16结束-------------------------------------------------')

    # data = scio.loadmat(r'D:\程序\Meta-gae-pytorch\data\sim1.mat')
    # dataset = np.array(data['ts'])
    # for i in range(0,50):
    #     output_name = "result/sim/sim1/" + str(i)+ ".txt"
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    # print('----------------------------------------------sim1结束-------------------------------------------------')
    # data = scio.loadmat(r'D:\程序\Meta-gae-pytorch\data\sim2.mat')
    # dataset = np.array(data['ts'])
    # n_nodes = 10
    # for i in range(0,50):
    #     output_name = "result/sim/sim2/"+str(i)+".txt"
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    # print('----------------------------------------------sim2结束-------------------------------------------------')
    # data = scio.loadmat(r'D:\程序\Meta-gae-pytorch\data\sim3.mat')
    # dataset = np.array(data['ts'])
    # n_nodes = 15
    # for i in range(0,50):
    #     output_name = "result/sim/sim3/"+str(i)+".txt"
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    # print('----------------------------------------------sim3结束-------------------------------------------------')
    # data = scio.loadmat(r'D:\程序\Meta-gae-pytorch\data\sim8.mat')
    # dataset = np.array(data['ts'])
    # n_nodes = 5
    # for i in range(0,50):
    #     output_name = "result/sim/sim4/"+str(i)+".txt"
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    # print('----------------------------------------------sim4结束-------------------------------------------------')
    # data = scio.loadmat(r'D:\程序\Meta-gae-pytorch\data\sim10.mat')
    # dataset = np.array(data['ts'])
    # n_nodes = 5
    # for i in range(0,50):
    #     output_name = "result/sim/sim5/"+str(i)+".txt"
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    # print('----------------------------------------------sim5结束-------------------------------------------------')
    # data = scio.loadmat(r'D:\程序\Meta-gae-pytorch\data\sim14.mat')
    # dataset = np.array(data['ts'])
    # n_nodes = 5
    # for i in range(0,50):
    #     output_name = "result/sim/sim6/"+str(i)+".txt"
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    # print('----------------------------------------------sim6结束-------------------------------------------------')
    # data = scio.loadmat(r'D:\程序\Meta-gae-pytorch\data\sim15.mat')
    # dataset = np.array(data['ts'])
    # n_nodes = 5
    # for i in range(0,50):
    #     output_name = "result/sim/sim7/"+str(i)+".txt"
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    # print('----------------------------------------------sim7结束-------------------------------------------------')
    # data = scio.loadmat(r'D:\程序\Meta-gae-pytorch\data\sim16.mat')
    # dataset = np.array(data['ts'])
    # n_nodes = 5
    # for i in range(0,50):
    #     output_name = "result/sim/sim8/"+str(i)+".txt"
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    # print('----------------------------------------------sim8结束-------------------------------------------------')
    # data = scio.loadmat(r'D:\程序\Meta-gae-pytorch\data\sim22.mat')
    # dataset = np.array(data['ts'])
    # n_nodes = 5
    # for i in range(0,50):
    #     output_name = "result/sim/sim9/"+str(i)+".txt"
    #     X = dataset[(0+i*200):(200+i*200),:]
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    # print('----------------------------------------------sim8结束-------------------------------------------------')
    # path = r'data/RestingState_MTL/individual_left_mtl_reduced'
    # files = os.listdir(path)
    # true_dag = np.loadtxt('data/RestingState_MTL/sub1_target.csv', delimiter=',')
    # result = np.zeros_like(true_dag)
    # for file in files:
    #     f_name = str(file)
    #     tr = '/'
    #     filename = path + tr + f_name
    #     output_name = "result/real/left/" + f_name
    #     X = np.loadtxt(filename)
    #     n = MetaRLEC()
    #     n.learn(X)
    #     np.savetxt(output_name, n.causal_matrix)
    #     result[n.causal_matrix != 0] += 1
    # np.savetxt("result/left.txt",result)
    # print('----------------------------------------------left结束-------------------------------------------------')
    path = r'data/RestingState_MTL/individual_right_mtl_reduced'
    files = os.listdir(path)
    true_dag = np.loadtxt('data/RestingState_MTL/sub1_target.csv', delimiter=',')
    result = np.zeros_like(true_dag)
    for file in files:
        f_name = str(file)
        tr = '/'
        filename = path + tr + f_name
        output_name = "result/real/right/" + f_name
        X = np.loadtxt(filename)
        n = MetaRLEC()
        n.learn(X)
        np.savetxt(output_name, n.causal_matrix)
        result[n.causal_matrix != 0] += 1
    np.savetxt("result/right.txt",result)
    print('----------------------------------------------right结束-------------------------------------------------')


