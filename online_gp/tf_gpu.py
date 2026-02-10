import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
a = tf.constant([1.0])
print(a < 2.0)   # triggers a GPU Less op like your crash