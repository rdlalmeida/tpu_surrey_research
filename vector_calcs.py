import tensorflow as tf
import tenseal as ts
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

# TensorFlow setup complete

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

context.generate_galois_keys()
context.global_scale = 2**40

# TenSEAL setup complete

vx = [1, 2, 3, 4, 5]
vy = [5, 4, 3, 2, 1]

print("Original vectors:")
print("Vx = [" + str(vx)[1:-1] + "]")
print("Vy = [" + str(vy)[1:-1] + "]")

tfx = tf.convert_to_tensor(vx)
tfy = tf.convert_to_tensor(vy)

print("\n\nTensorFlow converted vectors:")
print("TF-x = " + str(tfx)[1:-1] + "]")
print("TF-y = " + str(tfy)[1:-1] + "]")

# Encrypt the vectors
enc_x = ts.ckks_vector(context, vx)
enc_y = ts.ckks_vector(context, vy)

print("\n\nEncrypted vectors: ")
print("Enc_x = " + str(enc_x)[1:-1] + "]")
print("Enc_y = " + str(enc_y)[1:-1] + "]")

# Add the two encrypted vectors and decrypt the result
enc_add = enc_x + enc_y

vadd = enc_add.decrypt()

print("Decrypted Result: ")
print("Dec_z = [" + str(vadd)[1:-1] + "]")


# Repeat the process for the dot product of the two vectors
enc_dp = enc_x.dot(enc_y)

# Decrypt and print the result to confirm
dp = enc_dp.decrypt()

print("Dot product x.y = " + str(dp)[1:-1])

print("-------------------------- TPU SCRIPT END ----------------------------------------")