import tensorflow as tf
import os

print("------------------------- TPU SCRIPT START ---------------------------------------")
print("TensorFlow version " + tf.__version__)

# Try and read the environment variables and set them if they are empty
try:
    val = os.environ["TPU_NAME"]
    if (val == ""):
        os.environ["TPU_NAME"] = "local"
        print("Set TPU_NAME = local")
except KeyError:
    os.environ["TPU_NAME"] = "local"
    print("Created TPU_NAME = local")

try:
    val = os.environ["NEXT_PLUGGABLE_DEVICE_USE_C_API"]
    if (val == ""):
        os.environ["NEXT_PLUGGABLE_DEVICE_USE_C_API"] = "true"
        print("Set NEXT_PLUGGABLE_DEVICE_USE_C_API = true")
except KeyError:
    os.environ["NEXT_PLUGGABLE_DEVICE_USE_C_API"] = "true"
    print("Created NEXT_PLUGGABLE_DEVICE_USE_C_API = true")

try:
    val = os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"]
    if (val == ""):
        os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = "/lib/libtpu.so"
        print("Set TF_PLUGGABLE_DEVICE_LIBRARY_PATH = /lib/libtpu.so")
except KeyError:
    os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = "/lib/libtpu.so"
    print("Created TF_PLUGGABLE_DEVICE_LIBRARY_PATH = /lib/libtpu.so")

@tf.function
def add_fn(x,y):
    z = x + y
    return z

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.TPUStrategy(cluster_resolver)

pi = 3.14159
gr = 1.61803

x = tf.constant(pi)
y = tf.constant(gr)
z = strategy.run(add_fn, args=(x,y))

print("TensolFlow calculation result: ")
print(z)

print("-------------------------- TPU SCRIPT END ----------------------------------------")