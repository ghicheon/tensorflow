from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./samples/MNIST_data', one_hot=True)

import tensorflow as tf
import math


LOAD_SAVED_FILE = False
#LOAD_SAVED_FILE = True


batch_size = 128
n_epochs=100



tf.reset_default_graph()

save_file = './train_model.ckpt'


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

######### W = tf.Variable(tf.zeros([784,10]))
######### b = tf.Variable(tf.zeros([10]))
######### y = tf.nn.softmax(tf.matmul(x,W) + b)
######### sess.run(tf.initialize_all_variables())

def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#######################################################################3
# 1st layer       conv2d -> rulu -> max_pool
#######################################################################3
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#######################################################################3
# Second layer        conv2d -> rulu -> max_pool
#######################################################################3
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#######################################################################3
# Densely Connected Layer      flat(784X1024) -> dropout -> flat(1024X10) -> softmax
#######################################################################3

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)






#######################################################################3
# Train and Evaluate the Model
#######################################################################3

############### original 
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
###############

cost =  tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_)) 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

if LOAD_SAVED_FILE == True:
	saver.restore(sess,save_file)
	test_acc = accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
	print("test accuracy %f" % (test_acc ) )

else:
	for epoch in range(n_epochs):
	   total_batch = math.ceil(mnist.train.num_examples/batch_size)

	   for i in range(total_batch):
		   batch = mnist.train.next_batch(batch_size)
		   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	   train_acc = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
	   test_acc = accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
	   print('-----------------------------------------------')
	   print("epoch %5d  =>  training accuracy: %5f , test accuracy:%5f" %
						   (epoch, train_acc,test_acc)) 
	   print('-----------------------------------------------')



	saver.save(sess,save_file)
	print('rained Model Saved........!')

