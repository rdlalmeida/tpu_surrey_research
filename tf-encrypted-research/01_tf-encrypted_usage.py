import tensorflow as tf
import tf_encrypted as tfe

@tfe.local_computation('input-provider')
def provide_input():
    # Normal TensorFlow operations cna be run locally
    # as par of defining a private input, in this
    # case on the machine of the input provider
    return tf.ones(shape=(5, 10))

# Build graph and run graph
@tfe.function
def matmul_func(x, w):
    y = tfe.matmul(x, w)
    return y.reveal().to_native()

if __name__ == "__main__":

    # Provide inputs
    w = tfe.define_private_variable(tf.ones(shape=(10, 10)))
    x = provide_input()

    # Eager execution
    y = tfe.matmul(x, w)
    res = y.reveal().to_native()
    
    res = matmul_func(x, w)

    print("Result = ", res)