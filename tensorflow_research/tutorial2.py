import torch
from torchvision import transforms
from random import randint
import pickle
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
from typing import Dict
import tenseal as ts
from time import time

def getContext(scheme_type, poly_modulus_degree, coeff_mod_bit_sizes, global_scale):
    if (str(scheme_type).lower() != "ckks" and str(scheme_type).lower() != "bfv"):
        raise Exception(f"Invalid scheme type provided: {scheme_type}. This function supports only 'CKKS' or 'BFV' schemes")
    else:
        scheme_type = ts.SCHEME_TYPE.CKKS if str(scheme_type).lower() == "ckks" else ts.SCHEME_TYPE.BFV

    allowed_degrees = [1024, 2048, 4096, 8192, 16384, 32768]
    if (poly_modulus_degree not in allowed_degrees):
        raise Exception(f"Invalid polynomial modulus degree provided: {poly_modulus_degree}. This function expects one of these values: {allowed_degrees}")
    
    # Build and return the required context
    context = ts.context(scheme_type, poly_modulus_degree, coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    context.global_scale = global_scale
    context.generate_galois_keys()
    return context

def decrypt(enc):
    return enc.decrypt().tolist()

# ----------------------------------- EXAMPLES ----------------------------------------------------------

# These "example" function are nothing but a simple encapsulation of a bunch of calculations
# used as example but that do not possess any long term meaning. Abstracting them in a function
# prevents the main function from getting loaded with irrelevant output.
def getPlainTensors(v1, v2):
    plain1 = ts.plain_tensor(v1, v2)

    print(f"First tensor: Shape = {{{plain1.shape}}} Data = {{{plain1.tolist()}}}")

    plain2 = ts.plain_tensor(np.array(v1)).reshape(v2)

    print(f"Second tensor: Shape = {{{plain2.shape}}} Data = {{{plain2.tolist()}}}")

    return (plain1, plain2)

def getEncryptedTensors(context, v1, v2):
    plain1 = ts.plain_tensor(v1, v2)
    plain2 = ts.plain_tensor(np.array(v1)).reshape(v2)
    encrypted_tensor1 = ts.ckks_tensor(context, plain1)
    encrypted_tensor2 = ts.ckks_tensor(context, plain2)

    print(f"Shape = {{{encrypted_tensor1.shape}}}")
    print(f"Encrypted Data = {{{encrypted_tensor1}}}")

    encrypted_tensor_from_np = ts.ckks_tensor(context, np.array(v1).reshape(v2))
    print(f"Shape = {{{encrypted_tensor_from_np.shape}}}")

    return (encrypted_tensor1, encrypted_tensor2)

def roundTensor(tensor):
    for i in range(0, len(tensor)):
        if (isinstance(tensor[i], list)):
            tensor[i] = roundTensor(tensor[i])
        else:
            tensor[i] = round(tensor[i], 0)
    
    return tensor

if __name__ == "__main__":

    # Use the following dictionary to trigger the calculation on and off, given that these can take a lot of
    # time sometimes
    switches = {
        "add": True,
        "sub": True,
        "mul": True,
        "mpt": True,
        "neg": True,
        "pwr": True,
        "pol": True,
        "sig": True
   }

    scheme_type = "CKKS"
    poly_modulus_degree = 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    global_scale = 2 ** 40
    context = getContext(
        scheme_type=scheme_type,
        poly_modulus_degree=poly_modulus_degree, 
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        global_scale=global_scale
        )
    
    v1 = [1, 2, 3, 4]
    v2 = [2, 2]
    
    # Get the plain representation of the vectors (tensors)
    (plain_v1, plain_v2) = getPlainTensors(v1=v1, v2=v2)
    plain_v1_l = plain_v1.tolist()
    plain_v2_l = plain_v2.tolist()

    # Grab the encrypted version of those tensors
    (enc_v1, enc_v2) = getEncryptedTensors(context=context, v1=v1, v2=v2)

    # Adding two encrypted tensors
    if (switches["add"]):
        start_t = time()
        add_enc_v = enc_v1 + enc_v2
        add_dec_v = decrypt(add_enc_v)
        end_t = time()

        fmt_list = roundTensor(decrypt(add_enc_v))

        print("Adding two encrypted tensors: ")
        print(f"Plain equivalent {plain_v1_l} + {plain_v2_l}\nDecrypted result: {add_dec_v}\nApproximate result: {fmt_list}\nOperation took {(end_t - start_t) * 1000} milliseconds\n\n")
    
    # Subtracting two encrypted tensors
    if (switches["sub"]):
        start_t = time()
        sub_enc_v = enc_v1 - enc_v2
        end_t = time()
        
        fmt_list = roundTensor(decrypt(sub_enc_v))
        
        print("Subtracting two encrypted tensors: ")
        print(f"Plain equivalent {plain_v1_l} - {plain_v2_l}\nDecrypted result: {decrypt(sub_enc_v)}\nApproximate result: {fmt_list}\nOperation took {(end_t - start_t) * 1000} milliseconds\n\n")

    # Multiplying two encrypted tensors
    if (switches["mul"]):
        start_t = time()
        mul_enc_v = enc_v1 * enc_v2
        end_t = time()

        fmt_list = roundTensor(decrypt(mul_enc_v))
        
        print("Multiplying two encrypted tensors: ")
        print(f"Plain equivalent {plain_v1_l} * {plain_v2_l}\nDecrypted result: {decrypt(mul_enc_v)}\nApproximate result: {fmt_list}\nOperation took {(end_t - start_t) * 1000} milliseconds\n\n")

    # Multiplying an encrypted tensor and a plain tensor
    if (switches["mpt"]):
        plain_tensor = ts.plain_tensor([5, 6, 7, 8], [2, 2])
        
        start_t = time()
        mpt_enc_v = enc_v1 * plain_tensor
        end_t = time()

        fmt_list = roundTensor(decrypt(mpt_enc_v))
        
        print("Multiplying an encrypted tensor and a plain tensor: ")
        print(f"Plain equivalent: {plain_v1_l} * {plain_tensor.tolist()}\nDecrypted result: {mpt_enc_v}\nApproximate result: {fmt_list}\nOperation took {(end_t - start_t) * 1000} milliseconds\n\n")

    # Negating an encrypted tensor
    if (switches["neg"]):
        start_t = time()
        neg_enc_v = -enc_v1
        end_t = time()

        fmt_list = roundTensor(decrypt(neg_enc_v))

        print("Negating an encrypted tensor: ")
        print(f"Plain equivalent: -{plain_v1_l}\nDecrypted result: {decrypt(neg_enc_v)}\nApproximate result: {fmt_list}\nOperation took {(end_t - start_t) * 1000} milliseconds\n\n")

    # Exponentiation of an encrypted tensor
    if (switches["pwr"]):
        start_t = time()
        pwr_env_v = enc_v1 ** 3
        end_t = time()

        fmt_list = roundTensor(decrypt(pwr_env_v))

        print("Exponentiation of an encrypted tensor: ")
        print(f"Plain equivalent: {plain_v1_l} ^ 3\nDecrypt result: {decrypt(pwr_env_v)}\nApproximate result: {fmt_list}\nOperation took {(end_t - start_t) * 1000} milliseconds\n\n")

    # Polynomial evaluation of an encrypted tensor
    if (switches["pol"]):
        start_t = time()
        pol_enc_v = enc_v1.polyval([1, 0, 1, 1])
        end_t = time()

        fmt_list = roundTensor(decrypt(pol_enc_v))

        print("Polynomial evaluation of an encrypted tensor: ")
        print(f"X = {plain_v1_l}")
        print(f"1 + X^2 + X^3 = {decrypt(pol_enc_v)}")
        print(f"Approximate result: {fmt_list}")
        print(f"Operation took {(end_t - start_t) * 1000} milliseconds\n\n")

    # Sigmoid approximation of an encrypted tensor
    if (switches["sig"]):
        start_t = time()
        sig_enc_v = enc_v1.polyval([0.5, 0.197, 0, -0.004])
        end_t = time()

        print("Sigmoid approximation of an encrypted tensor: ")
        print(f"X = {plain_v1_l}")
        print(f"0.5 + 0.197 * X - 0.004 * X^3 = {decrypt(sig_enc_v)}\nOperation took {(end_t - start_t) * 1000} milliseconds\n\n")