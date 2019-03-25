import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets('/tmp/data/', one_hot = True ) # one hot means one is on others are off

n_classes = 10 
batch_size = 128  # tiles
hm_epochs = 10 # how many epoches do you want , cycles of feed forward and backprop
chunk_size = 28 
n_chunks = 28 
rnn_size = 128 

# height x width for matrix !!!!!!!!!!
x = tf.placeholder('float',[None,784]) # -> input data 
y = tf.placeholder('float') 

def recurrent_network_model(x):
    # neural network model
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])), # weight matrix for layer 1 
                        'biases':tf.Variable(tf.random_normal([n_classes]))} # biases matrix for layer 1 
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.rnn(lstm_cell,x,dtype=tf.float32)
    output = tf.add(tf.matmul(outputs[-1],layer['weights']),layer['biases'])
    return output 

# specify how we want the data do in that model 
def train_neural_network(x):
    prediction = recurrent_network_model(x)
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