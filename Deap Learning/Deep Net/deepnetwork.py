import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot = True ) # one hot means one is on others are off
'''
we have 10 classes , 0-9
one_hot : 
0 = [1,0,0,0,0,0,0,0,0,0] 
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
and so on 
'''
# computation graph:

# model :
# 3 hidden layers, each one has 500 nodes 
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500 

n_classes = 10 
batch_size = 100  # tiles

# height x width for matrix !!!!!!!!!!
x = tf.placeholder('float',[None,784]) # -> input data 
y = tf.placeholder('float') 

def neural_network_model(data):
    # neural network model
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])), # weight matrix for layer 1 
                        'biases':tf.Variable(tf.random_normal([1,n_nodes_hl1]))} # biases matrix for layer 1 

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                        'biases':tf.Variable(tf.random_normal([1,n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                        'biases':tf.Variable(tf.random_normal([1,n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                    'biases':tf.Variable(tf.random_normal([1,n_classes]))}

    #input_data * weight + bias 
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # -> a pre-defined threshold / activation function 
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])

    return output 

# specify how we want the data do in that model 
def train_neural_network(x):
    prediction = neural_network_model(x)
    # when use tf.nn.softmax_cross_entropy_with_logits_v2, need to pay attention to the order of variables passed in ...
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,labels = y)) # y is the known label for training , label
    optimizer = tf.train.AdamOptimizer().minimize(cost) # learning_rate = 0.001 by default 
    hm_epochs = 10 # how many epoches do you want , cycles of feed forward and backprop
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c 
            print('epoch', epoch,'completed out of ', hm_epochs,'loss: ',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy: ',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)


'''
input > weighted (unique) > hidden layer 1 (activation function) > weights > hidden layer 2 > ... > output layer 

forward neural network 
------------------
compare output to intended output > cost or loss fuction (cross entropy : -> how close are we)
optimation function (optimizer) > minimize cost ( AdamOptimizer ... SGD, AdaGrad)
-> backpropagation 
------------------

feed forward + backprop = epoch 

'''