# 1-1 Neural Network Adder
[Inference](#inference)

If you're reading this, you've probably decided to pick up deep learning. We start off with a simple TensorFlow implementation of a neural network adder. We do this for 2 reasons:

1. It's really simple! It trains really fast.
2. You don't have to download other data sets like MNIST or CIFAR, or something else.

You can run it from your laptop within 10 seconds and see the results. Without further ado, let's begin.

## Inference
We start with inference (diagram to be inserted).
```python
def inference(data):
  weights = tf.get_variable('weights', [2, 1], tf.float32, tf.zeros_initializer)
  biases = tf.get_variable('biases', [1], tf.float32, tf.zeros_initializer)
  result = tf.matmul(data, weights) + biases
  return result
```

That's pretty much it. You're done for inference.

## Loss

We use simple L2 loss for this. It makes sense because: 

1. If the neural network outputs a 0, but the answer is 1, the loss will be 0.5. 
2. If the neural network outputs a 0, but the answer is 0, the loss will be 0. 
3. If the neural network outputs a 1, but the answer is 1, the loss will be 0.
4. And so on...

You can check out [this link](https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html) for more options. 

```python
# result: output of neural network
# gt: ground truth
def loss_fn(result, gt):
  return tf.nn.l2_loss(result - gt)
```

## Train

With the loss function defined, we need some kind of optimizer. We will use Adam Optimizer for this because it is pretty simple to use. You can check out [this link](https://www.tensorflow.org/versions/r0.11/api_docs/python/train.html#optimizers) for more options.

```python
def train(loss, global_step):
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
```

## Combined

With that, you are ready to write the train loop to feed in the data, get the loss, and tune the weights!

```python
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
```

See full code [here](adder1_1.py).

# Conclusion

As you can see, by Step 10000, the loss has approached 0.000000. While this is a good result (unless of course you code your loss function wrongly), you can't see the values of the weights and biases changing with each step. You can't see sample inputs and corresponding results as well! We will do this in [1-2 Neural Network Adder](../1-2-Neural-Network-Adder).
