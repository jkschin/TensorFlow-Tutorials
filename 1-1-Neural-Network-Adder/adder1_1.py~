import tensorflow as tf
import numpy as np

def inference(data):
  weights = tf.get_variable('weights', [2, 1], tf.float32, tf.zeros_initializer)
  biases = tf.get_variable('biases', [1], tf.float32, tf.zeros_initializer)
  result = tf.matmul(data, weights) + biases
  return result

# result: output of neural network
# gt: ground truth
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

if __name__ == '__main__':
  train_loop()