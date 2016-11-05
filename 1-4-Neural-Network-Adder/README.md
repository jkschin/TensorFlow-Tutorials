# 1-4 Neural Network Adder

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