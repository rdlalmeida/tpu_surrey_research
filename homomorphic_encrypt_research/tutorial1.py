import torch
import tenseal as ts
import pandas as pd
import random
from time import time

import numpy as np
import matplotlib.pyplot as plt

# Define the number of epochs for both plain and encrypted training
EPOCHS = 5

def split_train_test(x, y, test_ratio=0.3):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)

    # Delimiter between test and train data
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]

def heart_disease_data():
    data = pd.read_csv("./tensorflow_research/datasets/framingham.xls")

    # Drop rows with missing values
    data = data.dropna()

    # Drop some features
    data = data.drop(columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"])

    # Balance data
    grouped = data.groupby("TenYearCHD")
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73).reset_index(drop=True))

    # Extract labels
    y = torch.tensor(data["TenYearCHD"].values).float().unsqueeze(1)

    data = data.drop(columns=["TenYearCHD"])

    # Standardize data
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    return split_train_test(x, y)

def random_data(m=1024, n=2):
    # data separable by the line 'y=x'
    x_train = torch.randn(m, n)
    x_test = torch.randn(m // 2, n)
    y_train = (x_train[:, 0] >= x_train[:, 1]).float().unsqueeze(0).t()
    y_test = (x_test[:, 0] >= x_test[:, 1]).float().unsqueeze(0).t()
    return x_train, y_train, x_test, y_test

class LR(torch.nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)
    
    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out
    
def train(model, optim, criterion, x, y, epochs=EPOCHS):
    for e in range(1, epochs + 1):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        print(f"Loss at epoch {e}: {loss.data}")
    return model

def accuracy(model, x, y):
    out = model(x)
    correct = torch.abs(y - out) < 0.5
    return correct.float().mean()

class EncryptedLR:
    def __init__(self, torch_lr):
        # TenSEAL processes lists and not torch tensors,
        # so we take out the parameters from the PyTorch model
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()

        # We accumulate gradients and counts the number of iterations
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    def forward(self, enc_x):
        # We don't need to perform sigmoid as this model
        # will only be used for evaluation, and the label
        # can be deduced without applying sigmoid
        enc_out = enc_x.dot(self.weight) + self.bias
        enc_out = EncryptedLR.sigmoid(enc_out)
        return enc_out
    
    def backward(self, enc_x, enc_out, enc_y):
        out_minus_y = (enc_out - enc_y)
        self._delta_w += enc_x * out_minus_y
        self._delta_b += out_minus_y
        self._count += 1

    def update_parameters(self):
        if self._count == 0:
            raise RuntimeError("You should at least run one forward iteration")
        
        # Update weights
        # We use a small regularization term to keep the output
        # of the linear layer in the range of the sigmoid approximation
        self.weight -= self._delta_w * (1 /self._count) + self.weight * 0.05
        self.bias -= self._delta_b * (1 / self._count)
        # Reset gradient accumulators and iterations count
        self._delta_w = 0
        self._delta_b = 0
        self._count = 0

    @staticmethod
    def sigmoid(enc_x):
        # We use the polynomial approximation of degree 3
        # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
        # which fits the function pretty well in the range [-5, 5]
        return enc_x.polyval([0.5, 0.197, 0, -0.004])
    
    def plain_accuracy(self, x_test, y_test):
        # Evaluate accuracy of the model on
        # the plain (x_test, y_test) dataset
        w = torch.tensor(self.weight)
        b = torch.tensor(self.bias)
        out = torch.sigmoid(x_test.matmul(w) + b).reshape(-1, 1)
        correct = torch.abs(y_test - out) < 0.5
        return correct.float().mean()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    ################################################
    ## You can use the functions below to perform ##
    ## the evaluation with an encrypted model     ##
    ################################################

    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)

    def decrypt(self, context):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

def encrypted_evaluation(model, enc_x_test, y_test):
    t_start = time()

    correct = 0

    for enc_x, y in zip(enc_x_test, y_test):
        # Encrypted evaluation
        enc_out = model(enc_x)
        # Plain comparison
        out = enc_out.decrypt()
        out = torch.tensor(out)
        out = torch.sigmoid(out)

        if torch.abs(out - y) < 0.5:
            correct += 1
    t_end = time()
    print(f"Evaluated test_set of {len(x_test)} entries in {int(t_end - t_start)} seconds")
    print(f"Accuracy: {correct}/{len(x_test)} = {correct / len(x_test)}")
    return correct /len(x_test)

def plot_normal_dist(mean, var, rmin=-10, rmax=10):
    x = np.arange(rmin, rmax, 0.01)
    y = normal_dist(x, mean, var)
    fig = plt.plot(x, y)

def encrypted_out_distribution(eelr, enc_x_text):
    w = eelr.weight
    b = eelr.bias
    data = []
    for enc_x in enc_x_test:
        enc_out = enc_x.dot(w) + b
        data.append(enc_out.decrypt())

    data = torch.tensor(data)
    mean, var = map(float, [data.mean(), data.std() ** 2])
    plot_normal_dist(mean, var)
    print("Distribution on encrypted data:")
    plt.show()

# --------------------------------------- MAIN ------------------------------------------------
if __name__ == "__main__":
    torch.random.manual_seed(73)
    random.seed(73)

    # You can use whatever data you want without modification to the tutorial
    x_train, y_train, x_test, y_test = heart_disease_data()

    print("############# Data summary #############")
    print(f"x_train has shape: {x_train.shape}")
    print(f"y_train has shape: {y_train.shape}")
    print(f"x_test has shape: {x_test.shape}")
    print(f"y_test has shape: {y_test.shape}")
    print("#######################################")

    n_features = x_train.shape[1]
    model = LR(n_features)
    # Use gradient descent with a Leaning_rate=1
    optim = torch.optim.SGD(model.parameters(), lr=1)
    # Ise Binary Cross Entropy Loss
    criterion = torch.nn.BCELoss()

    model = train(model, optim, criterion, x_train, y_train)

    plain_accuracy = accuracy(model, x_test, y_test)
    print(f"Accuracy on plain test_set: {plain_accuracy}")

    eelr = EncryptedLR(model)

    # parameters
    # poly_mod_degree = 4096
    poly_mod_degree = 16384
    # coeff_mod_bit_sizes = [40, 20, 40]
    coeff_mod_bit_sizes = [60, 40, 40, 40, 60]
    # create TenSEALContext
    ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    # Scale of ciphertext to use
    ctx_eval.global_scale = 2 ** 20
    # This key is needed for doing dot-product operations
    ctx_eval.generate_galois_keys()

    t_start = time()
    enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in x_test]
    t_end = time()
    print(f"Encryption of the test-set took {int(t_end - t_start)} seconds")

    # Encrypt the model's parameters
    eelr.encrypt(ctx_eval)

    encrypted_accuracy = encrypted_evaluation(eelr, enc_x_test, y_test)
    diff_accuracy = plain_accuracy - encrypted_accuracy
    print(f"Difference between plain and encrypted accuracies: {diff_accuracy}")

    if diff_accuracy < 0:
        print("Oh! We got a better accuracy on the encrypted test-set! The noise was on our side...")

    # Parameters
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]

    # create TenSEALContext
    ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx_training.global_scale = 2 ** 21
    ctx_training.generate_galois_keys()

    t_start = time()
    enc_x_train = [ts.ckks_vector(ctx_training, x.tolist()) for x in x_train]
    enc_y_train = [ts.ckks_vector(ctx_training, y.tolist()) for y in y_train]
    t_end = time()
    print(f"Encryption of the training_set took {int(t_end - t_start)} seconds")

    normal_dist = lambda x, mean, var: np.exp(- np.square(x - mean) / (2 * var)) / np.sqrt(2 * np.pi * var)

    # Plain distribution
    lr = LR(n_features)
    data = lr.lr(x_test)
    mean, var = map(float, [data.mean(), data.std() ** 2])
    plot_normal_dist(mean, var)
    print("Distribution on plain data: ")
    plt.show()

    # Encrypted distribution
    eelr = EncryptedLR(lr)
    eelr.encrypt(ctx_training)
    encrypted_out_distribution(eelr, enc_x_train)

    eelr = EncryptedLR(LR(n_features))
    accuracy = eelr.plain_accuracy(x_test, y_test)
    print(f"Accuracy at epoch #0 is {accuracy}")

    times = []

    for epoch in range(EPOCHS):
        eelr.encrypt(ctx_training)

        # If you want to keep an eye on the distribution to make sure
        # the function approximation is still working fine
        # WARNING: this operation is time consuming
        # encrypted_out_distribution(eelr, enc_x_train)

        t_start = time()
        for enc_x, enc_y in zip(enc_x_train, enc_y_train):
            enc_out = eelr.forward(enc_x)
            eelr.backward(enc_x, enc_out, enc_y)
        
        eelr.update_parameters()
        t_end = time()
        times.append(t_end - t_start)

        eelr.decrypt()
        accuracy = eelr.plain_accuracy(x_test, y_test)
        print(f"Accuracy at epoch #{epoch + 1} is {accuracy}")

    print(f"\nAverage time per epoch: {int(sum(times) / len(times))} seconds")
    print(f"Final accuracy is {accuracy}")

    diff_accuracy = plain_accuracy - accuracy

    print(f"Difference between plain and encrypted accuracies: {diff_accuracy}")

    if diff_accuracy < 0:
        print("Oh! We got a better accuracy when training on encrypted data! The noise was on our side!")