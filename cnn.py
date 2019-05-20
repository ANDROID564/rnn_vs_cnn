# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 18:40:21 2018

@author: TANVEER_MUSTAFA
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:05:01 2018

@author: TANVEER_MUSTAFA
"""
import tensorflow as tf
#tf.reset_default_graph()
#import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist=input_data.read_data_sets("tmp/data/",one_hot=True)


hm_epochs = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float')
y = tf.placeholder('float')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#(input data * weight )+biases
#mqake 100 layers

def convolutional_neural_network(x):#, keep_rate):
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }
    
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


 # Reshape input to a 4D tensor 
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution Layer, using our function
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    # Convolution Layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)
    
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    
    output = tf.matmul(fc, weights['out']) + biases['out']
    return output
    



def train_neural_network(x):
    prediction=convolutional_neural_network(x)
    cost=tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)  )
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    #learnning rate =0.001
    
    #cycles feedforward+ backprop
#    hm_epochs=10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        #training of model
        for epoch in range(hm_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x ,epoch_y=mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _ ,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c #track each time
            print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss)
        
        
        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) #tell identical
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))        #of correctness
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))        


train_neural_network(x)


