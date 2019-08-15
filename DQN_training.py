from __future__ import print_function
from GazeboWorld import GazeboWorld
import csv
import tensorflow as tf
import random
import numpy as np
import time
import rospy
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = InteractiveSession(config=config)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from collections import deque
import keras as k
filename = "/home/mudassir/ros_ws/src/logsdqn.csv"
reward_file = open(filename, "w")
reward_writer=csv.writer(reward_file, delimiter=',')
reward_writer.writerow(['Step', 'Reward', 'Loss'])


GAME = 'GazeboWorld'
ACTIONS = 7 # number of valid actions
SPEED = 2 # DoF of speed
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10. # timesteps to observe before training
EXPLORE = 20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 20000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
MAX_EPISODE = 15000
MAX_T = 200
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
RGB_IMAGE_HEIGHT = 228
RGB_IMAGE_WIDTH = 304
CHANNEL = 3
TAU = 0.001 # Rate to update target network toward primary network
H_SIZE = 8*10*64
IMAGE_HIST = 4


def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
	tf.summary.scalar('mean', mean)
	with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev', stddev)
	tf.summary.scalar('max', tf.reduce_max(var))
	tf.summary.scalar('min', tf.reduce_min(var))
	tf.summary.histogram('histogram', var)
		
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial, name="weights")

def bias_variable(shape):
	initial = tf.constant(0., shape = shape)
	return tf.Variable(initial, name="bias")

def conv2d(x, W, stride_h, stride_w):
	return tf.nn.conv2d(x, W, strides = [1, stride_h, stride_w, 1], padding = "SAME")

def layer_norm(x,y):
        return tf.contrib.layers.layer_norm(x,activation_fn=y, trainable=True, begin_norm_axis=1, begin_params_axis=-1, center=True, scale=True)
        

class QNetwork(object):
	"""docstring for ClassName"""
	def __init__(self,sess):
		# network weights
		# input 128x160x1
		with tf.name_scope("Conv1"):
			W_conv1 = weight_variable([10, 14, IMAGE_HIST, 32])
			variable_summaries(W_conv1)
			b_conv1 = bias_variable([32])
		# 16x20x32
		with tf.name_scope("Conv2"):
			W_conv2 = weight_variable([4, 4, 32, 64])
			variable_summaries(W_conv2)
			b_conv2 = bias_variable([64])
		# 8x10x64
		with tf.name_scope("Conv3"):
			W_conv3 = weight_variable([3, 3, 64, 64])
			variable_summaries(W_conv3)
			b_conv3 = bias_variable([64])
		# 8x10x64
		with tf.name_scope("FCValue"):
			W_value = weight_variable([H_SIZE, 512])
			variable_summaries(W_value)
			b_value = bias_variable([512])
			# variable_summaries(b_ob_value)

		with tf.name_scope("FCAdv"):
			W_adv = weight_variable([H_SIZE, 512])
			variable_summaries(W_adv)
			b_adv = bias_variable([512])
			# variable_summaries(b_adv)

		with tf.name_scope("FCValueOut"):
			W_value_out = weight_variable([512, 1])
			variable_summaries(W_value_out)
			b_value_out = bias_variable([1])
			# variable_summaries(b_ob_value_out)

		with tf.name_scope("FCAdvOut"):
			W_adv_out = weight_variable([512, ACTIONS])
			variable_summaries(W_adv_out)
			b_adv_out = bias_variable([ACTIONS])
			# variable_summaries(b_ob_adv_out

                
		# input layer
		self.state = tf.placeholder("float", [None, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, IMAGE_HIST])
		# Conv1 layer
		h_conv1 = tf.nn.relu(conv2d(self.state, W_conv1, 8, 8) + b_conv1)
		# Conv2 layer
		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2, 2) + b_conv2)
		# Conv2 layer
		h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1, 1) + b_conv3)
		h_conv3_flat = tf.reshape(h_conv3, [-1, H_SIZE])

		# FC ob value layer
		#h_fc_value = tf.nn.relu(tf.matmul(h_conv3_flat, W_value) + b_value)
		#Q_value = tf.matmul(h_fc_value, W_value_out) + b_value_out

		# FC ob adv layer
		h_fc_adv = tf.nn.relu(tf.matmul(h_conv3_flat, W_adv) + b_adv)		
		advantage = tf.matmul(h_fc_adv, W_adv_out) + b_adv_out
		
		# Q = value + (adv - advAvg)
		#advAvg = tf.expand_dims(tf.reduce_mean(advantage, axis=1), axis=1)
		#advIdentifiable = tf.subtract(advantage, advAvg)
		self.readout = advantage




		# define the ob cost function
		self.a = tf.placeholder("float", [None, ACTIONS])
		self.y = tf.placeholder("float", [None])
		self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), axis=1)
		self.td_error = tf.square(self.y - self.readout_action)
		self.cost = tf.reduce_mean(self.td_error)
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars/2]):
		op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)

def trainNetwork():
	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
	#sess = tf.InteractiveSession()	
	#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
	with tf.name_scope("OnlineNetwork"):
		online_net = QNetwork(sess)
	with tf.name_scope("TargetNetwork"):
		target_net = QNetwork(sess)
	rospy.sleep(1.)

	reward_var = tf.Variable(0., trainable=False)
	reward_epi = tf.summary.scalar('reward', reward_var)
	loss_var = tf.Variable(0., trainable=False)
	loss_epi = tf.summary.scalar('loss', loss_var)
	# define summary
	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter('./logs', sess.graph)

	# Initialize the World
	env = GazeboWorld()
	print('Environment initialized')

	# Initialize the buffer
	D = deque()

	# get the first state 
	depth_img_t1 = env.GetDepthImageObservation()
	depth_imgs_t1 = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=2)
	terminal = False
	
	# saving and loading networks
	trainables = tf.trainable_variables()
	trainable_saver = tf.train.Saver(trainables)
	sess.run(tf.global_variables_initializer())
	checkpoint = tf.train.get_checkpoint_state("./saved_network")
	print('checkpoint:', checkpoint)
	if checkpoint and checkpoint.model_checkpoint_path:
		trainable_saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print("Could not find old network weights")
		
	# start training
	episode = 0
	epsilon = INITIAL_EPSILON
	r_epi = 0.
	t = 0
	T = 0
	rate = rospy.Rate(5)
	print('Number of trainable variables:', len(trainables))
	targetOps = updateTargetGraph(trainables,TAU)
	loop_time = time.time()
	last_loop_time = loop_time
	while episode < MAX_EPISODE and T<=500000:
		env.ResetWorld()
		t = 0
		r_epi = 0.
		loss=0
		terminal = False
		reset = False
		loop_time_buf = []
		action_index = 0
		while not reset and not rospy.is_shutdown():
			depth_img_t1 = env.GetDepthImageObservation()
			depth_img_t1 = np.reshape(depth_img_t1, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1))
			depth_imgs_t1 = np.append(depth_img_t1, depth_imgs_t1[:, :, :(IMAGE_HIST - 1)], axis=2)
			reward_t, terminal, reset = env.GetRewardAndTerminate(t)
			if t > 0 :
				D.append((depth_imgs_t, a_t, reward_t, depth_imgs_t1, terminal))
				if len(D) > REPLAY_MEMORY:
					D.popleft()
			depth_imgs_t = depth_imgs_t1

			# choose an action epsilon greedily
			a = sess.run(online_net.readout, feed_dict = {online_net.state : [depth_imgs_t1]})
			readout_t = a[0]
			a_t = np.zeros([ACTIONS])
			if episode <= OBSERVE:
				action_index = random.randrange(ACTIONS)
				a_t[action_index] = 1
			else:
				if random.random() <= epsilon:
					print("----------Random Action----------")
					action_index = random.randrange(ACTIONS)
					a_t[action_index] = 1
				else:
					action_index = np.argmax(readout_t)
					a_t[action_index] = 1
			# Control the agent
			env.Control(action_index)

			if episode > OBSERVE :
				# # sample a minibatch to train on
				minibatch = random.sample(D, BATCH)
				#print("mini:",minibatch)
				y_batch = []
				# get the batch variables
				depth_imgs_t_batch = [d[0] for d in minibatch]
				#print("State:",depth_imgs_t_batch)
				a_batch = [d[1] for d in minibatch]
				r_batch = [d[2] for d in minibatch]
				depth_imgs_t1_batch = [d[3] for d in minibatch]
				Q1 = online_net.readout.eval(feed_dict = {online_net.state : depth_imgs_t1_batch})
				Q2 = target_net.readout.eval(feed_dict = {target_net.state : depth_imgs_t1_batch})
				for i in range(0, len(minibatch)):
					terminal_batch = minibatch[i][4]
					# if terminal, only equals reward
					if terminal_batch:
						y_batch.append(r_batch[i])
					else:
						y_batch.append(r_batch[i] + GAMMA * np.argmax(Q2[i]))

				#Update the network with our target values.
				online_net.train_step.run(feed_dict={online_net.y : y_batch,
													online_net.a : a_batch,
													online_net.state : depth_imgs_t_batch })
				loss = sess.run(online_net.cost, feed_dict={online_net.y: y_batch, online_net.a : a_batch,online_net.state : depth_imgs_t_batch })
				updateTarget(targetOps, sess) # Set the target network to be equal to the primary network.

			r_epi = r_epi + reward_t
                        reward_writer.writerow([T, r_epi,loss])
                        reward_file.flush()
			t += 1
			T += 1
			last_loop_time = loop_time
			loop_time = time.time()
			loop_time_buf.append(loop_time - last_loop_time)
			rate.sleep()

			# scale down epsilon
			if epsilon > FINAL_EPSILON and episode > OBSERVE:
				epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		#  write summaries
		if episode > OBSERVE:
			summary_str = sess.run(merged_summary, feed_dict={reward_var: r_epi, loss_var:loss})
			summary_writer.add_summary(summary_str, episode - OBSERVE)
			

		# save progress every 500 episodes
		if (episode+1) % 200  == 0 :
			trainable_saver.save(sess, 'saved_network/test1' + GAME + '-dqn', global_step = episode)

		if len(loop_time_buf) == 0:
			print("EPISODE", episode, "/ REWARD", r_epi, "/ steps ", T)
		else:
			print("EPISODE", episode, "/ REWARD", r_epi, "/ steps ", T,
				"/ LoopTime:", np.mean(loop_time_buf), "/ loss", loss)

		episode = episode + 1	

def main():
	trainNetwork()

if __name__ == "__main__":
	main()

