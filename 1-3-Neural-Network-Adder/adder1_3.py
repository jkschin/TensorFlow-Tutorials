import tensorflow as tf
import numpy as np

def inference(data):
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

def train_loop():
  global_step = tf.Variable(0, name='global_step', trainable=False)
  data = tf.placeholder(tf.float32, [None, 2], name='data')
  gt = tf.placeholder(tf.float32, [None, 1], name='gt')
  result = inference(data)
  loss = loss_fn(result, gt)
  train_op = train(loss, global_step)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(10001):
      data_in = np.random.uniform(low=0.0, high=1.0, size=(10, 2)).astype(np.float32)
      gt_in = np.expand_dims(np.array([np.sum(pair) for pair in data_in]).astype(np.float32), axis=1)
      _, loss_val = sess.run([train_op, loss], feed_dict={data: data_in, gt: gt_in})
      if i % 1000 == 0:
        print "Step: %d, Loss: %f" %(i, loss_val)
        # We added the graph nodes using tf.add_to_collection just now.
        # To access them now, simply call tf.get_collection.
        print tf.get_collection('weights')[0].name
        print sess.run(tf.get_collection('weights')[0]) # Notice the weights value changing.
        print ""
        print tf.get_collection('biases')[0].name
        print sess.run(tf.get_collection('biases')[0]) # Notice the bias value changing.
        print ""
        print tf.get_collection('result')[0].name
        print data_in
        print sess.run(tf.get_collection('result')[0], feed_dict={data: data_in}) # Notice feed_dict is required here because result depends on data.
        print ""

if __name__ == '__main__':
  train_loop()



