# https://nlp.stanford.edu/pubs/glove.pdf
import tensorflow as tf
import numpy as np
import collections
import os

class GloVe:
	def __init__(self, sess, voca_size, embedding_size):
		self.sess = sess
		self.voca_size = voca_size
		self.embedding_size = embedding_size


		with tf.name_scope("placeholder"):
			self.i_word_idx = tf.placeholder(tf.int32, [None, 1], name="i_word_idx")
			self.k_word_idx = tf.placeholder(tf.int32, [None, 1], name="k_word_idx")
			self.target = tf.placeholder(tf.float32, [None, 1], name="target")
			self.weighting = tf.placeholder(tf.float32, [None, 1], name="weighting")
			self.lr = tf.placeholder(tf.float32, name="lr")

		with tf.name_scope("embedding_table"):
			self.i_word_embedding_table = tf.Variable(tf.random_normal([voca_size, embedding_size])) 
			self.k_word_embedding_table = tf.Variable(tf.random_normal([voca_size, embedding_size])) 
			self.i_word_bias = tf.Variable(tf.random_normal([voca_size]))
			self.k_word_bias = tf.Variable(tf.random_normal([voca_size]))

		with tf.name_scope("embedding_lookup"):
			self.i_embedding = tf.nn.embedding_lookup(self.i_word_embedding_table, self.i_word_idx) # [N, 1, self.embedding_size]
			self.i_bias_embedding = tf.nn.embedding_lookup(self.i_word_bias, self.i_word_idx) # [N, 1]
			self.k_embedding = tf.nn.embedding_lookup(self.k_word_embedding_table, self.k_word_idx) # [N, 1, self.embedding_size]
			self.k_bias_embedding = tf.nn.embedding_lookup(self.k_word_bias, self.k_word_idx) # [N, 1]

		with tf.name_scope('probability'):
			# dot product
			self.probability = tf.matmul(self.i_embedding, tf.transpose(self.k_embedding, [0, 2, 1])) # [N, 1, 1]
			self.probability = tf.reshape(self.probability, [-1, 1]) + self.i_bias_embedding + self.k_bias_embedding # [N, 1]

		with tf.name_scope("cost"):
			self.cost = self.weighting * ((self.probability - tf.log(self.target))**2) # [N, 1]
			self.cost = tf.reduce_sum(self.cost)

		with tf.name_scope('train'): 
			optimizer = tf.train.AdagradOptimizer(self.lr)
			self.minimize = optimizer.minimize(self.cost)

		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)

		self.sess.run(tf.global_variables_initializer())

