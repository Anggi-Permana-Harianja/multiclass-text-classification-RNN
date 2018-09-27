'''
import modules
'''
from kumparanian import ds
import numpy as np
import pandas as pd
import tensorflow as tf

def preprocessing_data(filename):
	#import dataset
	data_frame = pd.read_csv(filename)
	#clean null values if any in article content
	data_frame = data_frame[pd.notnull(data_frame['article_content'])]
	#use only article_topic and article_content columns
	cols = ['article_topic', 'article_content']
	data_frame = data_frame[cols]
	data_frame.columns = ['article_topic', 'article_content']

	#make categorization in article_topic, factorize is to encode categorical
	data_frame['article_category'] = data_frame['article_topic'].factorize()[0]

	#create a category2index from factorized article_category
	from io import StringIO
	#to create a category2index we must take only unique factorized article_content
	category_id_data_frame = data_frame[['article_topic', 'article_category']].drop_duplicates().sort_values('article_category')
	category_to_id = dict(category_id_data_frame.values)
	id_to_category = dict(category_id_data_frame[['article_category', 'article_topic']].values)
	data_frame = data_frame[0 : 5000]

	features = data_frame.article_content
	labels = data_frame.article_category
	labels = np.array([1 if label <= 14 else 0 for label in labels])

	from string import punctuation
	all_text = ''.join([char for char in features if char not in punctuation])
	reviews = all_text.split('\n')

	all_text = ' '.join(reviews)
	words = all_text.split()

	#encoding the words
	from collections import Counter
	counts = Counter(words)
	vocab = sorted(counts, key = counts.get, reverse = True)

	#vocab2int
	vocab2int = {word: ii for ii, word in enumerate(vocab, 1)}
	int2vocab = {ii: word for ii, word in enumerate(vocab, 1)}

	#encode into int
	article_int = []
	for review in reviews:
		article_int.append([vocab2int[word] for word in review.split()])

	#non-zero-index, this return an array of article_int for arranging the sequence
	non_zero_idx = [ii for ii, article in enumerate(article_int) if len(article) != 0]
	article_int = [article_int[ii] for ii in non_zero_idx]

	
	#arrange in sequence
	seq_len = 200
	features = np.zeros((len(article_int), seq_len), dtype = int)
	for i, row in enumerate(article_int):
		features[i, -len(row) : ] = np.array(row)[ : seq_len]

	return features, labels, id_to_category, vocab2int
	
def split_data(features, labels):
	#split data into train, validation, test dataset
	split_frac = 0.8
	split_idx = int(len(labels) * 0.8)
	
	train_x, val_x = features[ : split_idx], features[split_idx : ]
	train_y, val_y = labels[ : split_idx], labels[split_idx : ]

	test_idx = int(len(val_y) * 0.5)
	val_x, test_x = val_x[ : test_idx], val_x[test_idx : ]
	val_y, test_y = val_y[ : test_idx], val_y[test_idx : ]

	return train_x, val_x, test_x, train_y, val_y, test_y

def get_batch(x, y, batch_size = 100):
	n_batches = len(y) // batch_size
	x, y = x[ : n_batches * batch_size], y[ : n_batches * batch_size]
	for ii in range(0, len(y), batch_size):
		yield x[ ii : ii + batch_size], y[ ii : ii + batch_size]


#hyperparameter
lstm_size = 256
lstm_layers = 2
batch_size = 500
learning_rate = 0.001

#class of RNN model
class Model(object):
	def __init__(self, lstm_size, lstm_layers, batch_size, learning_rate, filename):
		self.lstm_size = lstm_size
		self.lstm_layers = lstm_layers
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		
		features, labels, self.id_to_category, vocab2int = preprocessing_data(filename)
		self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = split_data(features, labels)

		#build graph
		n_words = len(vocab2int) + 1
		self.graph = tf.Graph()

		#build placeholders
		with self.graph.as_default():
			self.inputs_ = tf.placeholder(tf.int32, [None, None], name = 'inputs')
			self.labels_ = tf.placeholder(tf.int32, [None, None], name = 'labels')
			self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

		#embedding layers
		embed_size = 500
		with self.graph.as_default():
			embed_weights = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
			embed_model = tf.nn.embedding_lookup(embed_weights, self.inputs_)

		#buid deepLSTM 
		with self.graph.as_default():
			def get_a_cell(lstm_size, keep_prob):
				lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
				lstm_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = self.keep_prob)
				return lstm_cell_dropout

			with tf.name_scope('lstm'):
				self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(lstm_size, self.keep_prob) for _ in range(lstm_layers)])

			self.initial_state = self.stacked_cell.zero_state(batch_size, tf.float32)

		#RNN forward pass
		with self.graph.as_default():
			self.outputs, self.final_state = tf.nn.dynamic_rnn(self.stacked_cell, embed_model, 
													 initial_state = self.initial_state)

		#RNN output, we connect it to a Fully Connected NN
		with self.graph.as_default():
			#get predictions, only take the last cell outputs as our target
			self.predictions = tf.contrib.layers.fully_connected(self.outputs[:, -1], 1, activation_fn = tf.sigmoid)
			#get cost
			self.cost = tf.losses.mean_squared_error(self.labels_, self.predictions)
			#optimizer
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

		#RNN validation if needed
		with self.graph.as_default():
			self.correct_pred = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), self.labels_)
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		
	#training method
	def train(self):
		epochs = 4

		with tf.Session(graph = self.graph) as sess:
			tf.global_variables_initializer().run()
			iteration = 1

			for epoch in range(epochs):
				#get initial state
				state = sess.run(self.initial_state)

				for ii, (x, y) in enumerate(get_batch(self.train_x, self.train_y, self.batch_size), 1):
					feed = {self.inputs_: x, 
							self.labels_: y[:, None], 
							self.keep_prob : 0.5, 
							self.initial_state: state}

					loss, state, _ = sess.run([self.cost, self.final_state, self.optimizer], feed_dict = feed)

					if iteration % 2 == 0:
						print('epoch:{}/{}, iteration:{}, loss: {}'.format(epoch, epochs, iteration, loss))

					if iteration % 5 == 0:
						val_acc = []
						#set initial validation state
						val_state = self.stacked_cell.zero_state(self.batch_size, tf.float32)

						for x, y in get_batch(self.val_x, self.val_y, self.batch_size):
							feed = {self.inputs_: x, 
									self.labels_: y[ :, None], 
									self.keep_prob: 1, 
									self.initial_state: state}
							batch_acc, state = sess.run([self.accuracy, self.final_state], feed_dict = feed)
							val_acc.append(batch_acc)
							print(batch_acc)

						print('val_acc: {}'.format(np.mean(val_acc)))

					iteration += 1


if __name__ == '__main__':
    # NOTE: Edit this if you add more initialization parameter
    model = Model(lstm_size = lstm_size, lstm_layers = lstm_layers, batch_size = batch_size, 
    	          learning_rate = learning_rate, filename = 'data.csv')

    # Train your model
    model.train()

    # Save your trained model to model.pickle
    #model.save()
