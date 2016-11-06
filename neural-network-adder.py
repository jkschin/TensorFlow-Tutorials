import tensorflow as tf
import numpy as np

def inference(data):
	with tf.name_scope('inference'):
		with tf.variable_scope('hidden1'):
			weights = tf.get_variable('weights', [2, 1], tf.float32, tf.zeros_initializer)
			biases = tf.get_variable('biases', [1], tf.float32, tf.zeros_initializer)
			result = tf.matmul(data, weights) + biases

			# You can see tf.add_to_collection as adding these graph nodes for easy access later.
			tf.add_to_collection('weights', weights)
			tf.add_to_collection('biases', biases)
			tf.add_to_collection('result', result)
	return result

def loss_fn(result, gt):
	return tf.nn.l2_loss(result - gt)

def train(loss, global_step):
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

def train_loop(continue_training):
	global_step = tf.Variable(0, name='global_step', trainable=False)
	data = tf.placeholder(tf.float32, [None, 2], name='data')
	gt = tf.placeholder(tf.float32, [None, 1], name='gt')
	result = inference(data)
	loss = loss_fn(result, gt)
	train_op = train(loss, global_step)
	saver = tf.train.Saver()
	tf.add_to_collection('data', data) # This adds the entry point for the graph.
	with tf.Session() as sess:
		if continue_training:
			print "Resuming training from previous restore point"
			saver.restore(sess, 'train_dir/my-model-10000')
		else:
			print "Starting training from scratch"
			sess.run(tf.initialize_all_variables())
		for i in range(10001):
			data_in = np.random.uniform(low=0.0, high=1.0, size=(10, 2)).astype(np.float32)
			gt_in = np.expand_dims(np.array([np.sum(pair) for pair in data_in]).astype(np.float32), axis=1)
			_, loss_val = sess.run([train_op, loss], feed_dict={data: data_in, gt: gt_in})
			if i % 1000 == 0:
				print "Step: %d, Loss: %f" %(i, loss_val)
				# We added the graph nodes using tf.add_to_collection just now.
				# To access them now, simply call tf.get_collection.
				print tf.get_collection('weights')[0].name # Notice that the name of this node is hidden1/weights:0, without the inference.
				print sess.run(tf.get_collection('weights')[0]) # Notice the weights value changing.
				print ""
				print tf.get_collection('biases')[0].name # Notice that the name of this node is hidden1/biases:0, without the inference.
				print sess.run(tf.get_collection('biases')[0]) # Notice the bias value changing.
				print ""
				print tf.get_collection('result')[0].name # Notice that the name of this node is inference/hidden1/add:0, with the inference.
				print data_in
				print sess.run(tf.get_collection('result')[0], feed_dict={data: data_in})
				print ""
				saver.save(sess, 'train_dir/my-model', global_step=i)

def eval():
	with tf.Session() as sess:
		# Note that when saver is called, it is unnecessary to call tf.initialize_all_variables as it will wipe all values to 0.
		saver = tf.train.import_meta_graph('train_dir/my-model-10000.meta')
		saver.restore(sess, 'train_dir/my-model-10000')
		data = tf.get_collection('data')[0]
		result = tf.get_collection('result')[0]
		# This step prints all the variables that are saved in the graph.
		for var in tf.all_variables():
			print var.name, sess.run(var)
		print ""
		print "Result: ", sess.run(result, feed_dict={data: np.array([[100, 100]])}) # Notice that the answer is very close to adding the numbers!

if __name__ == '__main__':
	train_loop(True)
	eval()



