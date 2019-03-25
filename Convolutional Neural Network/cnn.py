import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot = True ) # one hot means one is on others are off

n_classes = 10 
batch_size = 128  # tiles

# height x width for matrix !!!!!!!!!!
x = tf.placeholder('float',[None,784]) # -> input data 
y = tf.placeholder('float') 

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME') # moving one by one 

def maxpooled2d(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME') # moving 2 by 2 


def convolutional_neural_network(x):
    # neural network model
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])), # weight matrix for layer 1 
                'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
                'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
                'out':tf.Variable(tf.random_normal([1024,n_classes]))} # biases matrix for layer 1 

    biases = {'B_conv1':tf.Variable(tf.random_normal([32])), # weight matrix for layer 1 
                'B_conv2':tf.Variable(tf.random_normal([64])),
                'B_fc':tf.Variable(tf.random_normal([1024])),
                'out':tf.Variable(tf.random_normal([n_classes]))} # biases matrix for layer 1 

    x = tf.reshape(x,shape = [-1,28,28,1])

    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['B_conv1'])
    conv1 = maxpooled2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['B_conv2'])
    conv2 = maxpooled2d(conv2)

    fc = tf.reshape(conv2,[-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['B_fc'])

    out = tf.add(tf.matmul(fc,weights['out']),biases['out'])

    return out

# specify how we want the data do in that model 
def train_neural_network(x):
    prediction = convolutional_neural_network(x)
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
