import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F 
from collections import namedtuple
import numpy as np
from intestine_environment import IntestineEnvironment
import matplotlib.pyplot as plt
import os

#simple linear NN with ReLU activation function
class LinearNet(nn.Module):
    def __init__(self, observation_size, hidden_size, n_actions):
        super(LinearNet, self).__init__()
        self.linearnet = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        logits = self.linearnet(x)
        return logits

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
    
def iterate_batches(environment, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    observation = environment.reset()
    #softmax to turn logits into probabilities
    softmax = nn.Softmax(dim=-1)
    while True:
        #change observation to tensor
        observation_tensor = torch.tensor(observation, dtype = torch.float)
        #get probabilities for each action
        action_probabilities_tensor = softmax(net(observation_tensor))
        #convert probabilities into numpy array
        action_probabilities = action_probabilities_tensor.detach().numpy()
        #take a random action sampled from a distribution according to the action probabilities
        action = np.random.choice(len(action_probabilities), p = action_probabilities)
        next_observation, reward, is_done = environment.step(action)
        #add the step reward into our total episode reward
        episode_reward += reward
        #append our step into namedtuple and list
        step = EpisodeStep(observation = observation, action = action)
        episode_steps.append(step)
        #check if the episode have concluded
        if is_done:
            #add whole episode into namedtuple and list
            e = Episode(reward = episode_reward, steps = episode_steps)
            batch.append(e)
            #reset reward and steps
            episode_reward = 0.0
            episode_steps = []
            #reset environment
            next_observation = environment.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        #make the observation our current one
        observation = next_observation

def filter_batch(batch, percentile):
    #create list of rewards
    rewards = list(map(lambda s: s.reward, batch))
    #grab 'elite' episodes within the top percentile 
    reward_bound = np.percentile(rewards, percentile)
    #find the mean of the rewards (monitoring only)
    reward_mean = float(np.mean(rewards))
    train_observations = []
    train_actions = []
    #for every episode in the batch
    for reward, steps in batch:
    #check if the episode has higher total reward than our boundary
        if reward < reward_bound:
            continue
        #if it does, populate the lists of observations and actions we will train on
        train_observations.extend(map(lambda step: step.observation, steps))
        train_actions.extend(map(lambda step: step.action, steps))
    #convert obs and acts into tensors
    train_observations_tensor = torch.FloatTensor(train_observations)
    train_actions_tensor = torch.LongTensor(train_actions)
    return train_observations_tensor, train_actions_tensor, reward_bound, reward_mean

def save(model, file_name = 'model.pt'):
    model_folder_path = './model'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
        
    file_name = os.path.join(model_folder_path, file_name)
    torch.save(model.state_dict(), file_name)

if __name__ == "__main__":
    HIDDEN_SIZE = 64
    BATCH_SIZE = 8
    PERCENTILE = 70

    Vmax , Vmin = 0.1, 0.01
    input_file = r"C:\Users\gould\Desktop\PhD\pipe-flow\pass_to_python\intestine_flow.lmp"

    environment = IntestineEnvironment(input_file, Vmax, Vmin)
    
    observation_size = 1
    num_actions = 6

    #initialise our net, loss, and optimizer
    net = LinearNet(observation_size, HIDDEN_SIZE, num_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = net.parameters(), lr = 0.005)

    #create lists for useful variables to track
    loss_list, reward_mean_list, rw_bound_list = [], [], []

    #perform main training loop
    for iter_no, batch in enumerate(iterate_batches(environment, net, BATCH_SIZE)):
        #get our batch
        observation_tensor, action_tensor, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)
        #reshape observations and actions to fit into the net and objective functions respectively
        obs_size = observation_tensor.size(dim = 0)
        observation_tensor = observation_tensor.view(obs_size, 1)
        #zero gradients
        optimizer.zero_grad()
        #pass observations through network
        action_scores_tensor = net(observation_tensor)
        #get loss
        loss_tensor = objective(action_scores_tensor, action_tensor)
        #back propogation
        loss_tensor.backward()
        #apply weight updates
        optimizer.step()

        #print out results
        print("%d: loss = %.3f, reward_mean=%.1f, rw_bound=%.1f" % (iter_no, loss_tensor.item(), reward_mean, reward_bound))
        #append to lists
        loss_list.append(loss_tensor.item())
        reward_mean_list.append(reward_mean)
        rw_bound_list.append(reward_bound)

        if iter_no >= 24:
            print("Training complete")
            break 

    #save the model
    save(net)

    # let us make a simple graph to plot the reward over time
    fig = plt.figure(figsize=[7,5])
    ax = plt.subplot(111)
    x_data = np.arange(len(reward_mean_list))
    l = ax.fill_between(x_data, reward_mean_list)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Agent Reward per Batch')

    # set the limits
    ax.set_xlim(0, max(x_data))
    ax.set_ylim(0, 1.2*max(reward_mean_list))

    ax.grid('on')
    # change the fill into a blueish color with opacity .3
    l.set_facecolors([[.5,.5,.8,.3]])

    # change the edge color (bluish and transparentish) and thickness
    l.set_edgecolors([[0, 0, .5, .3]])
    l.set_linewidths([3])

    # remove tick marks
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8,.8,.8))
    ax.spines['top'].set_color((.8,.8,.8))

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    xlab.set_style('italic')
    xlab.set_size(10)
    ylab.set_style('italic')
    ylab.set_size(10)

    # tweak the title
    ttl = ax.title
    ttl.set_weight('bold')
    #save figure
    fig.savefig('reward_over_time.png')