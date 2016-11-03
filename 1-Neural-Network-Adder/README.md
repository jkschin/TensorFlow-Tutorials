# Neural Network Adder

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

