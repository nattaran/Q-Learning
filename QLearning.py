# The code is  for training and testing the Qlearning algorthm
# Nasrin Attaran - Feb 21#
# Trading session all sessions 
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import sklearn
import seaborn as sns
#import tensorflow
#import graphviz
import pydot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
###
import keras
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import Initializer
from keras.models import model_from_json
from keras import optimizers
# training set 1 - Monday Feb 26 - continous values
#data = pd.read_csv("training_data_26Feb.csv")
data = pd.read_csv("training_.csv")
training_data1 = data[['NP10_Class', 'NC4_Class', 'FC10_Class', 'FP1_Class', 'BidAskRatio_Class', 'TotalMarketDepth_Class', 'Price_Class','Price_dif' ]]


# testing set 1 -Tuesday Feb 27- continous values
#data = pd.read_csv("testing_data_27Feb.csv")
data = pd.read_csv("testing_.csv")
testing_data1 = data[['NP10_Class', 'NC4_Class', 'FC10_Class', 'FP1_Class', 'BidAskRatio_Class', 'TotalMarketDepth_Class', 'Price_Class','Price_dif' ]]


#testing set 2 - Thursday March 1

data = pd.read_csv("testing_2.csv")
testing_data2 = data[['NP10_Class', 'NC4_Class', 'FC10_Class', 'FP1_Class', 'BidAskRatio_Class', 'TotalMarketDepth_Class', 'Price_Class','Price_dif' ]]


class environment:
    def __init__(self):
        self.cost = 3.03    # setting the cost amount

    def partial_reward_cal (self, partial_reward, state, next_state ):
        if ( state[0] == 0 and state[1] == 1  and state[2] == 0   ): # if the agent in long position
            reward_temp = partial_reward
            if (next_state[0] ==  0 and next_state[1] ==  0 and next_state[2] ==  1):      # if the agent decides to close the long
                cost_temp = -self.cost                                                          # he should pay the $3 fee
            else:
                cost_temp = 0        # if the agent stays in the long state, he does not need to pay the $3 fee

        elif (state[0] == 1 and state[1] == 0  and state[2]==0): # if the agent is in short position
            reward_temp = -partial_reward                        # we reverse the partial_reward because the negative price differenc means positive reward
            if (next_state[0] ==  0 and next_state[1] ==  0 and next_state[2] ==  1  ):   # if the agent decides to close the short
                cost_temp = -self.cost                                                         # he should pay the $3 fee
            else:
                cost_temp = 0     # if the agent stays in the long state, he does not need to pay the $3 fee

        elif (state[0] == 0 and state[1] == 0 and state[2] ==1 ):  # if the agent is in flat state
           reward_temp = 0                                         # there is no reward
           if ((next_state[0] ==  0 and next_state[1] ==  1 and next_state[2] ==  0) or (next_state[0] == 1 and next_state[1]==0 and next_state[2]==0)):  # If the agent wants to open long or short position, he should pay the transaction fee ($3)
               cost_temp = -self.cost
           else:
               cost_temp = 0

        # the total reward is summation of partial reward and transaction fees
        return  (reward_temp +cost_temp)


# define a class for DQNAgent
class DQNAgent:
    def __init__(self, state_size, action_size, feature_num):
        self.state_size = state_size     # Number of variable we need to represent the agent state
                                         # -- the last 3 variable shows the agent state-  001: (flat), 010 (Long), 100 (Short)
                                         # In this code the state has 4 variable - 1 variable for logreturn + 3 variable for state
        self.action_size = action_size   # We have 3 actions  - flat (0) - buy (1) - sell (2)
        self.feature_num =feature_num    # number of market environment which I used - in this code, we have only one variable - logreturn_10S
        self.memory = deque(maxlen=50000) # Memory to store the samples- each sample is <state, next_state, acion, reward, fail, success> # save the 100 Episods of the data
        self.gamma = 0.95 #  # discount rate
        #self.epsilon = 1.0  # exploration rate - at the beginiing it is 1- after each step of training it will be reduced by epsilon_decay
        self.epsilon = 1.0  # exploration rate - at the beginiing it is 1- after each step of training it will be reduced by epsilon_decay

        self.epsilon_min = 0.01     #  the minimum value for exploration rate- It mean that in all steps of training there is possibility that the agent chooses the action by random - The randomness rate will decrease but it doesn't reach to 0
        self.epsilon_decay = 0.995  #0.995
        self.learning_rate = 0.00025  #0.001
        self.tau = .125

        # creating main model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()

        # initialize target model
        self.update_target_model()


    # Neural Net for Deep-Q learning Model using Keras
    def _build_model(self):

        model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu',  kernel_initializer='he_uniform'))   # bias_initializer=keras.initializers.Constant(value=0.5)
        # model.add(Dense(24, activation='relu',   kernel_initializer='he_uniform'))
        # model.add(Dense(self.action_size, activation='linear',  kernel_initializer='he_uniform'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #opt = RMSprop(lr=0.00025)
        #opt = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)
        #model.compile( loss ='mse',optimizer = opt  ) #keras.optimizers.RMSprop(lr=0.00025)
        model.add(Dense(64, input_dim=self.state_size, activation='relu', bias_initializer=keras.initializers.Constant(value=0.5))) #24 ,bias_initializer=keras.initializers.Constant(value=2)
        model.add(Dense(64, activation='relu', bias_initializer=keras.initializers.Constant(value=0.5)))  #, bias_initializer=keras.initializers.Constant(value=0.5)
       # model.add(Dense(64, activation='relu', bias_initializer=keras.initializers.Constant(value=0.5)))
        model.add(Dense(32, activation='relu', bias_initializer=keras.initializers.Constant(value=0.5)))
        model.add(Dense(16, activation='relu', bias_initializer=keras.initializers.Constant(value=0.5)))
        model.add(Dense(8, activation='relu', bias_initializer=keras.initializers.Constant(value=0.5)))
        model.add(Dense(self.action_size, activation='linear')) #, bias_initializer=keras.initializers.Constant(value=0.5)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # next_state_ function :
    #this function determines the next state based on the current state and the action
    #
    def next_state_ (self, state, action ):
        if (state[0] == 0 and state[ 1] == 0 and state[2] == 1 and action == 1):  # go to long state
            next_state = [0, 1, 0]
            #next_state[0, feature_num:feature_num + 3] = [0, 1, 0]  # go to the long state
        elif (state[0] == 0 and state[1] == 0 and state[2] == 1 and action == 2):  # go to long state
            next_state = [1, 0, 0]  # go to short state
        elif (state[0] == 0 and state[1] == 0 and state[2] == 1 and action == 0):
            next_state = [0, 0, 1]  # stay in the flat state

        # if the current state is long
        elif (state[0] == 0 and state[1] == 1 and state[2] == 0 and action == 2):  # if the agent is in the long and wants to close
            next_state = [0, 0, 1]
        elif (state[0] == 0 and state[1] == 1 and state[2] == 0 and action == 0):
            next_state = [0, 1, 0]

        # if the current state is short
        elif (state[0] == 1 and state[1] == 0 and state[2] == 0 and action == 1):
            next_state = [0, 0, 1]
        elif (state[0] == 1 and state[1] == 0 and state[ 2] == 0 and action == 0):
            next_state = [1, 0, 0]
        return next_state

    # Remember function :
    # the memory stores the agent's status for each experince
    # - Neural network needs needs to train with different agent previous experinces as well


    def remember(self, state, action, reward, next_state, success, fail):
        self.memory.append((state, action, reward, next_state, success, fail))
    # empty_mem function
    # We use this function to empty the memory from previous experience to add the current period experience to the memory

    def empty_mem (self) :
        self.memory.clear()

    # This function finds returns the list of correct action for each step
    # for example when the agent in the flat state, it can select among all actions{hold, buy and sell}
    # when it is in long state, it can just choose between {hold, sell}



    def correct_act(self, state):
        if (state[0] == 0) and (state[ 1] == 0)and (state[ 2] == 1):  # If the agent in hold state
            action_list = [0, 1, 2]

        elif (state[0] == 0) and (state[ 1] == 1)and  (state[ 2] == 0):  # if the agent is in long state
            action_list = [2, 0]

        elif (state[ 0] == 1) and (state[ 1] == 0)and  (state[2] == 0) :  # if the agent is in short state
            action_list = [1, 0]

        return (action_list)

    # check_act : this function check the action based on the current state
    # In each state, there are some forbidden action
    # for example in long state, the action "buy" is an incorrect action

    def check_act (self, state, action):
        if ((state[0]==0) and (state[1]==1) and (state[2]==0) and  (action == 1) ):
            action_status = False
        elif ((state[0]==1) and (state[1]==0) and (state[2]==0) and ((action == 2) )):
            action_status=  False
        else :
            action_status= True
        return action_status

    # act : This function determins the agent's action for each state
    # Based on exploration rate (epsilon) this action selection can be random or based on the prediction result of neural network
    # at beginning steps of training, almost all actions will be choosen randomely
    # after decresing the exploration rate, the agent uses the neural network to find the action

    def act(self, state ,  feature_num    ):
        if np.random.rand() <= self.epsilon: # choosing random actions - exploration phase
            #print ("Use Random Action")
            state_= state[0, feature_num :feature_num+3]  # this three variable shows the state's position (flat, long and short)
            action_list = self.correct_act(state_)
            action =  (random.choice(action_list ))
        # use the neural network to select the appropriat action
        else:
            print ("Use Neural Network")
            #self.counter = self.counter+1
            #print (self.counter)
            act_values = self.model.predict(state)    # using the Neural Network for making decisions
            action_value =np.argsort(act_values[0])   # sort (ascending)  the action based on the score values

            if (self.check_act (state [0,feature_num :feature_num+3 ], action_value[2] ) == True ):    # if the highest score is the correct action
                action= action_value[2]
            else:                                # if the highest score is not the correct action, the next highest action will be choosen
                action= action_value[1]
        return action

    # detect_state : retun the state position in form of string {"short", "long" and "flat"}
    def detect_state (self, state):
        if (state[0, feature_num] == 1 ):
            state="Short"
        elif (state[0, feature_num+1]==1):
            state="Long"
        elif (state[0, feature_num+2]==1) :
            state= "Flat"
        return state
    # Return the Neural network output value for correct action

    def action_predict (self, state , feature_num ) :
        act_values = self.model.predict(state)  # using the Neural Network for making decisions
        #print ("act_values",act_values)
        action_value = np.argsort(act_values[0])  # sort (ascending) the action based on the score
        if (self.check_act(state [0,feature_num :feature_num+3 ], action_value[2]) == True): # if the highest score belongs to correct action
            return action_value[2]   # return the highest score
        else:   # if the highest score is not correct action, return the second action
            return action_value[1]

    # replay : The main goal of this function is to train the neural network with random samples in the memory
    # after each episode of training, we choose the 32 samples randomly from memory ( The samples form previous episode can be inside the rendom selection)
    # Calculates the target reward based of MDP and train the network
    def replay(self, sample_size, batch_size):    # batch size is 32
    #def replay(self):
        minibatch = random.sample(self.memory, sample_size)
        #print ("inside replay")
        #print ("run the training step")
        minibatch_state= []
        minibatch_target=[]
        for state, action, reward, next_state, success, fail in minibatch:
            # set the reward value for trainig DQN
            # if (reward > 0 ):
            #     reward = 100
            # elif (reward <0):
            #     reward = -10
            # else:
            #     reward =0

            target = self.model.predict(state)
            target_next = self.model.predict (next_state)
            target_val  = self.target_model.predict(next_state)
            if (success == True  or fail == True  ) :
                target[0][action] = reward
            else:
                a= self.action_predict( next_state, feature_num)
                #a= np.argmax (target_next)
                #print ("state","next_state" , "action", state, next_state, a)
                target[0][action] = reward + self.gamma*target_val[0][a]
            minibatch_state.append (state[0])
            minibatch_target.append(target[0])

        minibatch_state = np.array( minibatch_state)
        minibatch_target= np.array(minibatch_target)

        #print ("state", minibatch_state)
        #print ("target", minibatch_target)

        self.model.fit(minibatch_state, minibatch_target, epochs=3,batch_size = batch_size, verbose=0)  # train the neural network
        #print ("epsilon", self.epsilon)

        #print("NN output", target_f)

        if self.epsilon > self.epsilon_min:
           self.epsilon *= self.epsilon_decay

    # def target_train(self):
    #     weights = self.model.get_weights()
    #     target_weights = self.target_model.get_weights()
    #     for i in range(len(target_weights)):
    #         target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
    #     self.target_model.set_weights(target_weights)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
# training function : Use different agent function to train the agent
def training (feature_num , reward_limit , risk_limit):

    out_df = pd.DataFrame(columns=[ 'Episode', 'TotalReward'])  # output datafram to save the total reward of all episodes
    Episod_len = 1024
    EPISODE  = 3000
    #NN_flag = False
    indx = 0
    #df_list = [training_data1, training_data2, training_data3]    # list of training dataframes
    df_list = [training_data1 ] # , training_data2
    df_index = 0
    for df in df_list:
        df_index +=1
        data_size = len(df.index)    # length of training datafram
        #print ("data" , df)
        for e in range(EPISODE):
            # if (e < 800) :
            #     NN_flag = False
            # else:
            #     NN_flag = True
            print ("Episod", e)
            success = False   # one flag to show when the agen reaches to +$20 profit
            fail = False      # one flag to show when the agen reaches to -$20 loss
            #agent.empty_mem() # clear the memory from last experiences
            total_reward = 0   # We use this parameter to calculate the total reward - add partial rewards in agent's state transitions
            # initiate the intial state  - the initial state should be flat
            index = random.randint(0,data_size-Episod_len-1)   # choose the initial state from random location
            #print ("the start point of training", index)
            # set the intial state of the agent to flat state
            Internal_state = np.array(df.iloc[index,0:feature_num]) # taking the i element in statet_mat as an initial state
            position_state = np.array([0,0,1])     # set the agent state to flat state
            Initial_state = np.concatenate((Internal_state, position_state), axis=0)
            #Initial_state[state_size-3:state_size ] =[0,0,1]  # set the initial state to flat (0)
            state = Initial_state    # at start of episod the state is equal to initial state
            state = np.reshape(state, [1, state_size ])
            # for each episode, the agent choose the actions until it reaches to max profit or max loss
            for time in range(Episod_len):
                par_reward = (df.loc[index, "Price_dif"] *1000)  # calculate the partial reward
                index += 1
                action = agent.act(state, feature_num)
                # generating the next state
                next_state = np.array(df.iloc[index, 0:feature_num])   # next row in the trining_data1 datafram can be considered as the next state
                pos_state = np.array([0,0,0])
                next_state = np.concatenate ((next_state , pos_state) , axis=0)
                next_state = np.reshape(next_state, [1, state_size ])
                next_state[0,feature_num:feature_num+3] = agent.next_state_(state[0, feature_num :feature_num+3] , action)
                reward = env.partial_reward_cal (par_reward, state[0,feature_num:feature_num+3], next_state[0,feature_num:feature_num+3] ) # finding the reward for the agent's action
                total_reward += reward #updating the total reward based on the partial reward
                # setting the succes and fail flags based on the obtained profit and loss
                if (total_reward >= reward_limit) :
                    success = True
                else:
                    success = False

                if (total_reward <= risk_limit) :
                    fail = True
                else:
                    fail = False

                # save the transaction information in the memory
                #Q_reward = agent.Reward(state, next_state)
                agent.remember(state, action, reward, next_state, success, fail)
                print (state[0,feature_num:feature_num+3],action, reward,total_reward)
                state = next_state     # the next state will be next current state

                # if the agent reaches to max profit or loss, it exits the current episode or training and will start the next episode
                if (success== True) or (fail == True):
                    break


            out_df.loc[indx] = pd.Series({ 'Episode': e, 'TotalReward': total_reward})
            indx += 1
            if  (len(agent.memory) > sample_size):
                agent.replay(sample_size, batch_size)  # batch_size
            #agent.target_train()
            agent.update_target_model()

        #if (len(agent.memory) > batch_size):    # after finishing each episode, we will train the neural network with 32 randon samples in memory
		 #   agent.replay(batch_size)   #batch_size

    out_df.to_csv('./results/April24_Training_Price.csv', index=False)
    agent.save("model.h5")

# testing

def testing (feature_num , reward_limit , risk_limit):
    agent.load("model.h5")
    testing_df = pd.DataFrame(columns=[ 'Episode', 'TotalReward'])  # output datafram to save the total reward of all episodes
    Episod_len = 1000
    EPISODE = 100
    indx = 0
    df_list = [testing_data1]  # list of testing dataframes
    df_index = 0
    for df in df_list:
        df_index+=1
        data_size = len(df.index)  # length of training datafram
        # print ("data" , df)
        for e in range(EPISODE):
            print("Episod", e)
            success = False  # one flag to show when the agen reaches to +$20 profit
            fail = False  # one flag to show when the agen reaches to -$20 loss
            total_reward = 0  # We use this parameter to calculate the total reward - add partial rewards in agent's state transitions
            # initiate the intial state  - the initial state should be flat
            index = random.randint(0, data_size - Episod_len - 1)  # choose the initial state from random location
            Internal_state = np.array( df.iloc[index, 0:feature_num])  # taking the i element in statet_mat as an initial state
            position_state = np.array([0, 0, 1])  # set the agent state to flat state
            Initial_state = np.concatenate((Internal_state, position_state), axis=0)
            # Initial_state[state_size-3:state_size ] =[0,0,1]  # set the initial state to flat (0)
            state = Initial_state  # at start of episod the state is equal to initial state
            state = np.reshape(state, [1, state_size])
            # for each episode, the agent choose the actions until it reaches to max profit or max loss
            for time in range(Episod_len):
                par_reward = (df.loc[index, "Price_dif"] * 1000)  # calculate the partial reward
                index += 1
                #action = agent.act(state, feature_num)
                # generating the next state
                act_values = agent.model.predict(state)
                # use the Neural Network for Action prediction
                action_value = np.argsort(act_values[0])  # sort the action based on the score [0]
                if (agent.check_act(state[0,feature_num :feature_num+3 ], action_value[2]) == True):
                    action= action_value[2]
                else:
                    action = action_value[1]
                         # determine the next_state based on the agent's current state and action
                next_state = np.array(df.iloc[index,0:feature_num])  # next row in the trining_data1 datafram can be considered as the next state
                pos_state = np.array([0, 0, 0])
                next_state = np.concatenate((next_state, pos_state), axis=0)
                next_state = np.reshape(next_state, [1, state_size])
                next_state[0, feature_num:feature_num + 3] = agent.next_state_(state[0, feature_num:feature_num + 3], action)

                # finding the reward for the agent's action
                reward = env.partial_reward_cal(par_reward, state[0, feature_num:feature_num + 3], next_state[0, feature_num:feature_num + 3])
                # updating the total reward based on the partial reward
                total_reward += reward
                # setting the succes and fail flags based on the obtained profit and loss
                if (total_reward >= reward_limit):
                    success = True
                else:
                    success = False

                if (total_reward <= risk_limit):
                    fail = True
                else:
                    fail = False

                print(state[0, feature_num:feature_num + 3], action, reward, total_reward)
                state = next_state  # the next state will be next current state
                # if the agent reaches to -20 or +20, it exits the current episode or training and will start the next episode
                if (success == True) or (fail == True):
                    break
                    # if (success == True):
                    #     result = 1
                    # else:
                    #     result = 0
            testing_df.loc[indx] = pd.Series({ 'Episode': e, 'TotalReward': total_reward})
            indx += 1
    testing_df.to_csv('./results/April25_Testing_Price_1march.csv', index=False)


# main function
if __name__ == "__main__":
    batch_size = 32
    sample_size= 1024
    feature_num = 7
    print("I am inside the main progran ")
    state_size = 10 # 6 vaiables for state representation - logreturn + 3 varaibles for state
    action_size = 3  # Buy, sell, hold
    # feature_num = 3 # we have 1 feature - logreturn
    reward = 0
    env = environment()
    agent = DQNAgent(state_size, action_size, feature_num)
    indx = 0
    reward_limit = 3000
    risk_limit = -3000
    training(feature_num, reward_limit, risk_limit)  #call training to train the agent
    #agent.load ("model.h5")
    testing(feature_num , reward_limit , risk_limit)


