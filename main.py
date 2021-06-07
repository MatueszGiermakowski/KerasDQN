'''
simple code using keras to train DQN on given openAI envs
'''
import argparse

import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
from keras.optimizers import Adam


'''
Below code is to train on nvidia GPU
'''
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DQN():

    def __init__(self, env, no_states, no_action, memory_size=100000, model_layers=[32, 64, 32],
                 batch_size=64, gamma=0.99, epsilon=[0.9, 0.01, 0.995]):
        """
        :param env: env variable maybe used later
        :param no_states: number of input we get from ennv
        :param no_action: nunmber of actions agentn can take
        :param memory_size: number of memories we store before deleting the last one (need to sample memory based on
        epsilon greedy policy)
        :param model_layers: number of neurons in each layer going from input to output,  there will be an additional
        output layer with amount of neurons being the same as desired output
        :param batch_size: number of
        :param gamma: discount factor for training the agent
        :param epsilon: factor for amount of times we take a random action rather than predicted one
         to explore the model first before exploring. Values are [starting_epsilon,min_epsilon,decay_rate]

        the metric we optimize the NN around will be mean squared error between
        our NN Q-table prediction and actual value of the Q-table

        adam uses the squared gradients to scale the learning rate,
        takes advantage of momentum by using moving average of the gradient instead of gradient itself
        """
        self.env = env
        self.states = no_states
        self.actions = no_action
        self.memsize = memory_size
        self.memory = []
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = self.generate_model(model_layers)
        self.model.compile(optimizer='Adam', loss='mse')
        self.model.summary()
        self.epsilon = epsilon

    def generate_model(self, model_layers):
        """
        :param model_layers: number of neurons in each layer going from input to output,
         there will be an additional output layer with amount of neurons being the same as desired output
        :return: returns keras model object
        activation function will be relu for all layers but the output one wheres its linear
        (selection of activation functions is inspired  by other ppl code but there doesnt seem to be that big of
        performence difference between different activation functions)
        """

        model = Sequential()
        model.add(Dense(units=model_layers[0], activation='relu', input_dim=self.states))
        if len(model_layers) > 1:
            for i in model_layers[1:]:
                model.add(Dense(units=i, activation='relu'))

        model.add(Dense(self.actions, activation='linear'))
        return model

    def remember(self, val):
        """
        :param val: val is condensed current_state + action + reward + next_state + done from env.step()
        :return:
        """
        self.memory.append(val)
        if len(self.memory) > self.memsize:
            del self.mem[0]

    def get_action(self, state):
        '''
        selecting a random action or predicted action based on epsilon greedy policy.
        1-epsilon precent times we choose the highest Q-value action and the other times we choose random action to
        explore the env
        ! keras predict expects 2 dimensional array while our state is in 1 dim so its necesecary to reform by using
        np.array(state).reshape(1, self.states) from [x] to [[x]]
        :param state: current inputs from env
        :return: value of the action (in cartpole case 0 or 1 )
        '''
        epsi = self.epsilon[0]
        if self.epsilon[0] > self.epsilon[1]:
            self.epsilon[0] *= self.epsilon[2]
        if np.random.random() < epsi:
            return self.env.action_space.sample()
        else:
            q_values = self.model.predict_on_batch(np.array(state).reshape(1, self.states))
            return np.argmax(q_values)

    def update_weights(self):
        '''
        we take batch_size from memory, update our prediction q-table and then update our weights based on mean
        square error between our q-table and actual q-values
        based on bellman equation
        Q(s’, a) = f(s’, θ), if s is not the terminal state (state at the last step)
        Q(s’, a) = r, if s is the terminal state
        also since we only know the value of 1 future action (because we can only take 1 action at each step), only
        1 action value will be updated
        ! keras predict expects 2 dimensional array while our state is in 1 dim so its necesecary to reform by using
        np.array(state).reshape(batch_size, state) from [x] to [[x]]
        also fit is called on whole batch to improve optimazation instead of doing it memory sample by sample
        '''

        current_state, reward, action, next_state, done = zip(*random.sample(self.memory, self.batch_size))
        pred_q = self.model.predict(np.array(current_state).reshape(self.batch_size, self.states))
        actual_q = self.model.predict(np.array(next_state).reshape(self.batch_size, self.states))
        for i in range(self.batch_size):
            k = int(action[i])
            if not done[i]:
                target = reward[i] + self.gamma * np.amax(actual_q[i][:])
            else:
                target = reward[i]

            pred_q[i][k] = target

        self.model.fit(np.array(current_state).reshape(self.batch_size, self.states), pred_q
                       , epochs=1, batch_size=1, verbose=0)


def run_dqn(render):
    '''
    :param render: if True the env will render
    runs the simulation connecting env and dqn
    '''
    env = gym.make('CartPole-v1')
    actions = env.action_space.n
    states = env.observation_space.shape[0]
    agent = DQN(env, states, actions)

    total_episodes = 500

    count = 0
    average_rewards = []
    for episodes in range(total_episodes):
        current_state = env.reset()
        total_reward = 0
        done = False
        while not done:
            if render:
                env.render()
            if count == 0:
                action = env.action_space.sample()
            else:
                action = agent.get_action(current_state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.remember([current_state, reward, action, next_state, done])
            current_state = next_state
            if count > agent.batch_size:
                agent.update_weights()
            count += 1
            if done:
                average_rewards.append(total_reward)
                break
        print("Episode: ", episodes, "Reward: ", total_reward, "Average: ", np.mean(average_rewards))
    agent.model.save("my_model")
    env.close()

def main(args):
    if args.no_render:
        run_dqn(False)
    else:
        run_dqn(True)


def parse_arguments():
    '''
    This function parses argumets from command line.

    Returns
    -------
        argparse.Namespace
        This is a namespace with attributes given by command line or their default
        values.
    '''
    parser = argparse.ArgumentParser(description='A simple example of DQN')

    parser.add_argument('--no_render',
                        action='store_true',
                        help='if set then the DQN agent will run')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
