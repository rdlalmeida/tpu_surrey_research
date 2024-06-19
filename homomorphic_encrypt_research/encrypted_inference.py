import tenseal as ts
from torchvision import transforms
from random import randint
from PIL import Image
from typing import Dict

# Create the TenSEAL context
def create_ctx():
    """Helper for creating the CKKS context.
    CKKS params:
        - Polynomial degree: 8192
        - Coefficient modulus size: [40, 21, 21, 21, 21, 21, 21, 40].
        - Scale: 2 ** 21
        - The setup requires the Galois keys for evaluating the convolutions."""
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    global_scale = 2 ** 21
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = global_scale
    ctx.generate_galois_keys()
    return ctx

# Sample an image
def load_input():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # The image sample folder has 350 sample images, named from "img_1.jpg" to "img_350.jpg"
    # Use the random integer generator to select a random image from this set
    idx = randint(1, 350)
    # Grab a random image from the data folder
    img_filename= f"data/mnist-samples/img_{idx}.jpg"
    print("Using " + img_filename + " as source file")
    img = Image.open(img_filename)
    return transform(img).view(28, 28).tolist(), img


# Helper for encoding the image
def prepare_input(ctx, plain_input):
    enc_input, windows_nb = ts.im2col_encoding(ctx, plain_input, 7, 7, 3)
    assert windows_nb == 64
    return enc_input

# Load a pretrained model and adapt the forward call for encrypted input
class ConvMNIST():
    """CANN for classifying MNIST data
    Input should be an encoded 28x28 matrix representing the image.
    TenSEAL can be used for encoding `tenseal.im2col_encoding(ctx, input_matrix, 7, 7, 3)`
    The input should also be normalized with a mean=0.1307 and an std=0.3081 before encryption.
    """

    def __init__(self, parameters: Dict[str, list]):
        self.conv1_weight = parameters["conv1_weight"]
        self.conv1_bias = parameters["conv1_bias"]
        self.fc1_weight = parameters["fc1_weight"]
        self.fc1_bias = parameters["fc1_bias"]
        self.fc2_weight = parameters["fc2_weight"]
        self.fc2_bias = parameters["fc2_bias"]
        self.window_nb = parameters["windows_nb"]
    
    def forward(self, enc_x: ts.CKKSVector) -> ts.CKKSVector:
        # Conv layer
        channels = []

        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, self.window_nb) + bias
            channels.append(y)
        
        out = ts.CKKSVector.pack_vectors(channels)

        # Squaring
        out.square_()
        # No need to flat
        # fc1_layer
        out = out.mm_(self.fc1_weight) + self.fc1_bias

        # Squaring
        out.square_()
        # Output layer
        out = out.mm_(self.fc2_weight) + self.fc2_bias
        return out
    
    @staticmethod

    def prepare_input(context: bytes, ckks_vector: bytes) -> ts.CKKSVector:
        try:
            ctx = ts.context_from(context)
            enc_x = ts.ckks_vector_from(ctx, ckks_vector)
        except:
            raise DeserializationError("Cannot deserialize context ot ckks_vector") # type: ignore
        
        try:
            _ = ctx.galois_keys()
        except:
            raise InvalidContext("The context provided doesn't hold any Galois keys")   # type: ignore
        
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == "__main__":
    print("OK!")