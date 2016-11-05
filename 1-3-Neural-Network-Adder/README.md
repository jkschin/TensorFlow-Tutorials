# 1-3 Neural Network Adder

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

Wait a minute, isn't that weird? Shouldn't the weights and biases have inference appended to it as well? In short, name_scopes are not appended to variables. But the add op has both name_scope and variable_scope appended. Do read the TensorFlow documentation for a greater elaboration. In [1-4 Neural Network Adder](../1-4-Neural-Network-Adder), we will talk about saving a model.