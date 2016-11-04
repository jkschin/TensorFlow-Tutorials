# 1-2 Neural Network Adder

If you've skipped [1-1 Neural Network Adder](../1-1-Neural-Network-Adder), do check it out first. So now you're thinking of printing the values of the weights and biases at every iteration, and see how they change. Perhaps you might even want to see the random data generated in each iteration and the result that the neural network gives. If you're impatient, you can skim through the first few parts, but I thought it would be useful to talk about this as most people would go through this. Coming from a Python background, most people might now do something like this:

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

You can now see the weights and biases change and be sure that the neural network adder is doing what it is supposed to do. You can also print the names of these nodes as well. While this is really nice, when you have very complex graphs, you might not want to call your variables 'weights1', 'weights2', and so on. That's what name scopes and variable scopes are for, and this will be in [1-3 Neural Network Adder](../1-3-Neural-Network-Adder).

