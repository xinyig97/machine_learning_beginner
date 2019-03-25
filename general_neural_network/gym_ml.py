import gym
import random 
import numpy as np 
import tflearn 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3 
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000 


def some_random_game():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
def initial_population():
    training_data = []
    scores  = []
    accepted_scores = []
    for _ in range(initial_games): # '_' is a dummy variable that just to show python that we are iterating throught the list and not care what the thing is 
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation,action])
            
            prev_observation = observation # kind of weirod 
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    print('avgerage accepted score: ', mean(accepted_scores))
    print('median accepted :', median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data
def neural_network_model(input_size):
    network = input_data(shape = [None,input_size,1], name = 'input')
    network = fully_connected(network,128,activation = 'relu')
    network = dropout(network,0.8) # not all neurals are active 

    network = fully_connected(network,256,activation = 'relu')
    network = dropout(network,0.8) # not all neurals are active 

    network = fully_connected(network,512,activation = 'relu')
    network = dropout(network,0.8) # not all neurals are active 

    network = fully_connected(network,256,activation = 'relu')
    network = dropout(network,0.8) # not all neurals are active 

    network = fully_connected(network,128,activation = 'relu')
    network = dropout(network,0.8) # not all neurals are active 

    network= fully_connected(network,2,activation= 'softmax')
    network = regression(network, optimizer='adam',learning_rate=LR, loss = 'categorical_crossentropy',name = 'targets')
    model = tflearn.DNN(network,tensorboard_dir='log')
    return model
def train_model(training_data, model  = False):
    X  = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)  # -> observations
    Y = np.array([i[1] for i in training_data])
    if not model:
        model = neural_network_model(input_size = len(X[0]))
    model.fit({'input':X},{'targets':Y}, n_epoch= 5, snapshot_step= 500, show_metric= True, run_id='openaistuff')
    return model

training_data = initial_population()
model = train_model(training_data)

# if you save the model and want to use later from loading, you need to have the exact same shape model skeleton to load  

scores = []
choices = []
for each_game in range(100):
    score = 0
    game_memory =[]
    pre_o = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(pre_o)==0:
            action = random.randrange(0,2)
        else:
           # action = np.argmax(model.predict(pre_o.reshape(-1,len(pre_o),1))[0])
        choices.append(action)
        new_ob, reward, done, info  = env.step(action)
        pre_o = new_ob
        game_memory.append([new_ob,action])
        score += reward
        if done:
            break
    scores.append(score)

print('aveger:' , sum(scores)/len(scores))