---
layout: page
title: Neural Network Adder
permalink: /neural-network-adder/
---

# **Neural Network Adder**

1. [Inference](#inference)
2. [Loss](#loss)
3. [Train](#train)
4. [Train Loop](#train-loop)
5. [Printing Values](#printing-values)
6. [Name Scopes and Variable Scopes](#name-scopes-and-variable-scopes)
7. [Saving a Model](#saving-a-model)
8. [Loading a Model](#loading-a-model)


If you're reading this, you've probably decided to pick up deep learning. We start off with a simple TensorFlow implementation of a neural network adder. We do this for 2 reasons:

1. It's really simple! It trains really fast.
2. You don't have to download other data sets like MNIST or CIFAR, or something else.

You can run it from your laptop within 10 seconds and see the results. Without further ado, let's begin.

# **Inference**

We start with inference (diagram to be inserted).

```python
def inference(data):
  weights = tf.get_variable('weights', [2, 1], tf.float32, tf.zeros_initializer)
  biases = tf.get_variable('biases', [1], tf.float32, tf.zeros_initializer)
  result = tf.matmul(data, weights) + biases
  return result
```

That's pretty much it. You're done for inference.

# **Loss**

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

# **Train**

With the loss function defined, we need some kind of optimizer. We will use Adam Optimizer for this because it is pretty simple to use. You can check out [this link](https://www.tensorflow.org/versions/r0.11/api_docs/python/train.html#optimizers) for more options.

```python
def train(loss, global_step):
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
```

# **Train Loop**

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

As you can see, by Step 10000, the loss has approached 0.000000. While this is a good result (unless of course you code your loss function wrongly), you can't see the values of the weights and biases changing with each step. You can't see sample inputs and corresponding results as well!

# **Printing Values**

So now you're thinking of printing the values of the weights and biases at every iteration, and see how they change. Perhaps you might even want to see the random data generated in each iteration and the result that the neural network gives. If you're impatient, you can skim through the first few parts, but I thought it would be useful to talk about this as most people would go through this. Coming from a Python background, most people might now do something like this:

```python
for i in range(10001):
  ...
  ...
  print (weights)
  print (biases)
  print (result)
```

You then realize that this doesn't actually work because these variables are not local to the function train_loop. You then decide to return these variables:

```python
def inference(data):
  ...
  ...
  return result, weights, biases

def train_loop():
  for i in range(10001):
    ...
    ...
    print (weights)
    print (biases)
    print (result)
```

You do get some results! However, they look weird and you have no idea what it is.

Tensor("add:0", shape=(?, 1), dtype=float32)

Well, that's because this variable is actually something like a node in the graph and you have to run it to get the values. Specifically, this is an add operation that outputs a size of (?, 1), where ? is the batch_size inferred, and the dtype is float32. You then call sess.run() like so:

```python
_, loss_val, weights_val, biases_val, result_val = sess.run([train_op, loss, weights, biases, result])
print loss_val
print weights_val
print biases_val
print result_val
```

Voila! It works! But this is a really ugly way to do it. I had to walk you through this because you will see the beauty of tf.add_to_collection() only after this.

You should actually modify inference and train_loop to the following:

```python
def inference(data):
  weights = tf.get_variable('weights', [2, 1], tf.float32, tf.zeros_initializer)
  biases = tf.get_variable('biases', [1], tf.float32, tf.zeros_initializer)
  result = tf.matmul(data, weights) + biases

  # You can see tf.add_to_collection as adding these graph nodes for easy access later.
  tf.add_to_collection('weights', weights)
  tf.add_to_collection('biases', biases)
  tf.add_to_collection('result', result)
  return result

def train_loop():
  ...
  ...
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
```

You can now see the weights and biases change and be sure that the neural network adder is doing what it is supposed to do. You can also print the names of these nodes as well. While this is really nice, when you have very complex graphs, you might not want to call your variables 'weights1', 'weights2', and so on. That's what name scopes and variable scopes are for.

# **Name Scopes and Variable Scopes**

This is a really short section. Name scopes are variable scopes are extremely useful because it really makes everything much cleaner. All you have to do is to add 2 lines and you're done!

```python
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
```

There are important things to note, however. You can see that the names of your nodes are now the following:

```python
hidden1/weights:0
hidden1/biases:0
inference/hidden1/add:0
```

Wait a minute, isn't that weird? Shouldn't the weights and biases have inference appended to it as well? In short, name_scopes are not appended to variables. But the add op has both name_scope and variable_scope appended. Do read the TensorFlow documentation for a greater elaboration. 

# **Saving a Model**

The first thing you have to do is to create a saver. After creating a saver, simply call it when you want the model to be saved. You can do it like this:

```python
def train_loop():
  ...
  ...
  train_op = train(loss, global_step)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(10001):
      ...
      if i % 1000 == 0:
        ...
        ...
        saver.save(sess, 'train_dir/my-model', global_step=i)
```

And you're done! The results will be written to train_dir. There are three interesting things in train_dir

1. checkpoint. This is a record of all the latest checkpoints.
2. my-model-XXXX. These are the weights at that point in time.
3. my-model-XXXX.meta. This is the entire graph definition that you can load in future for inference or re-training.

# **Loading a Model** 

To be continued...