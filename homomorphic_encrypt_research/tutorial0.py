import tenseal as ts
import time

doit_flag = False

def getBFVContext(mod_degree, plain_mod):
    # Build and return the BFV (Brakerski/Fan-Vercauteren) context
    return ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=mod_degree, plain_modulus=plain_mod)

if (__name__ == "__main__"):

    print("Getting a BFV context: ")
    context = getBFVContext(4096, 1032193)

    if (doit_flag):

        print("This context is " + ("private" if context.is_private() else "public"))

        sk = context.secret_key()
        context.make_context_public()

        print("After dropping the secret key, the context is now " + ("private" if context.is_private() else "public"))

        try:
            another_sk = context.secret_key()
            print("Got another secret key from the context!")
        except:
            print("Unable to get a secret key from a public context. Got an error instead.")

        plain_vector = [60, 66, 73, 81, 90]
        another_vector = [1, 2, 3, 4, 5]

        encrypted_vector = ts.bfv_vector(context, plain_vector)

        print("We just encrypted out plaintext vector of size: ", encrypted_vector.size())

        add_result = encrypted_vector + another_vector
        print("Added vector: [" + str(add_result._decrypt())[1:-1] + "]\n")

        sub_result = encrypted_vector - another_vector
        print("Subtracted vector: [" + str(sub_result._decrypt())[1:-1] + "]\n")

        mul_result = encrypted_vector * another_vector
        print("Multiplied vector: [" + str(mul_result._decrypt())[1:-1] + "]\n")

        encrypted_add = add_result + sub_result
        print("Encrypted added vector: [" + str(encrypted_add._decrypt())[1:-1] + "]\n")

        encrypted_sub = encrypted_add - encrypted_vector
        print("Encrypted subtracted vector: [" + str(encrypted_sub._decrypt())[1:-1] + "]\n")

        encrypted_mul = encrypted_add * encrypted_sub
        print("Encrypted multiplication vector: [" + str(encrypted_mul._decrypt())[1:-1] + "]\n")

        t_start = time.time()
        _ = encrypted_add * encrypted_mul
        t_end = time.time()
        print("c2c multiply time: {} ms".format((t_end - t_start) * 1000))

        t_start= time.time()
        _ = encrypted_add * another_vector
        t_end = time.time()
        print("c2p multiply time: {} ms".format((t_end - t_start) * 1000))
    
    print("Automatic relinearization is: " + ("on" if context.auto_relin else "off"))
    print("Automatic rescaling is: " + ("on" if context.auto_rescale else "off"))
    print("Automatic modulus switching is: " + ("on" if context.auto_mod_switch else "off"))

    try:
        print("global_scale: ", context.global_scale)
    except ValueError:
        print("The global_scale isn't defined yet")

    context.global_scale = 2 ** 20
    print("global_scale: ", context.global_scale)