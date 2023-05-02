# Neural Network model + Training code

from sklearn.model_selection import train_test_split
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class PUFWeightsPredictor(nn.Module):

    def __init__(self, input_dims=64, output_dims=1, weights_init=None):
        super(PUFWeightsPredictor, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.fc = nn.Linear(input_dims + 1, 1, bias=False)
        self.training = True
        if isinstance(weights_init, tuple):
            nn.init.uniform_(
                self.fc.weight, a=weights_init[0], b=weights_init[1])
        elif weights_init is not None:
            nn.init.normal_(self.fc.weight, mean=0.0, std=weights_init)

    def forward(self, phi):
        return self.fc(phi)

    def get_weights(self):
        state = self.state_dict().copy()
        wt = state["fc.weight"]
        wt = wt.reshape(self.input_dims + 1, 1).cpu().numpy()
        return wt


class Trainer:
    def __init__(self, model, num_epochs, lr, batch_size, train_ds, val_ds):
        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True
        )
        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=True
        )

    def train(self):
        criterion = nn.BCEWithLogitsLoss()
        # optimizer = Adam(self.model.parameters(), lr=self.lr)
        optimizer = SGD(self.model.parameters(), lr=self.lr)

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(device)

        for epoch in range(1, self.num_epochs + 1):
            running_train_loss = 0.0
            running_val_loss = 0.0

            # Training Loop
            self.model.train()
            for train_batch in self.train_dl:
                inputs, outputs = train_batch
                inputs = inputs.to(device)
                outputs = outputs.to(device)
                optimizer.zero_grad()
                predicted_outputs = self.model(inputs)
                predicted_outputs = predicted_outputs.to(device)
                train_loss = criterion(predicted_outputs, outputs)
                train_loss.backward()
                optimizer.step()

                running_train_loss += train_loss.item()

            # Calculate training loss value
            train_loss_value = running_train_loss/len(self.train_dl)

            # Validation Loop
            with torch.no_grad():
                self.model.eval()

                for val_batch in self.val_dl:
                    inputs, outputs = val_batch
                    inputs = inputs.to(device)
                    outputs = outputs.to(device)
                    predicted_outputs = self.model(inputs)
                    predicted_outputs = predicted_outputs.to(device)
                    val_loss = criterion(predicted_outputs, outputs)
                    running_val_loss += val_loss.item()

            # Calculate validation loss value
            val_loss_value = running_val_loss/len(self.val_dl)

            # Print the statistics of the epoch
            print(
                f"Epoch {epoch}/{self.num_epochs} Avg. training loss: {train_loss_value:.4f} Avg. val loss: {val_loss_value:.4f}")


def train_and_eval(params):
    results = []
    lr = params["lr"]
    num_epochs = params["num_epochs"]
    weight_init = params["weight_init"]
    target_training_size = params["training_size"]

    def puf_query(c, w):
        n = c.shape[1]
        phi = np.ones(n+1)
        phi[n] = 1
        for i in range(n-1, -1, -1):
            phi[i] = (2*c[0, i]-1)*phi[i+1]

        r = (np.dot(phi, w) > 0)
        return r

    # Problem Setup
    target = 0.99  # The desired prediction rate
    n = 64  # number of stages in the PUF

    # Initialize the PUF
    np.random.seed(int(time.time()))
    data = np.loadtxt('./weight_diff.txt')
    w = np.zeros((n+1, 1))
    for i in range(1, n+2):
        randi_offset = np.random.randint(1, 45481)
        w[i-1] = data[randi_offset-1]

    # Syntax to query the PUF:
    c = np.random.randint(0, 2, size=(1, n))  # a random challenge vector
    r = puf_query(c, w)
    # you may remove these two lines

    # You can use the puf_query function to generate your training dataset
    # ADD YOUR DATASET GENERATION CODE HERE
    print(f"Generating training data of size {target_training_size}")
    training_size = target_training_size
    X = np.random.randint(0, 2, size=(training_size, n))
    y = np.zeros((training_size, 1))

    for i in range(training_size):
        y[i] = puf_query(X[i].reshape(1, -1), w)

    def calc_phi(select_bits):
        phi_vals = []
        for i in range(len(select_bits)):
            target_slice = select_bits[i:]
            zeros = [z for z in target_slice if z == 0]
            phi = 1 if len(zeros) % 2 == 0 else -1
            phi_vals.append(phi)
        return np.array(phi_vals + [1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train_phi = np.apply_along_axis(calc_phi, 1, X_train)
    X_test_phi = np.apply_along_axis(calc_phi, 1, X_test)
    print(f"Done!")

    w0 = np.zeros((n+1, 1))  # The estimated value of w.
    # Try to estimate the value of w here. This section will be timed. You are
    # allowed to use the puf_query function here too, but it will count towards
    # the training time.

    print(f"Starting training for {num_epochs} epochs.")
    print("-" * 100)
    train_dataset = TensorDataset(torch.from_numpy(X_train_phi[:training_size]).float(
    ), torch.from_numpy(y_train[:training_size]).float())
    val_dataset = TensorDataset(torch.from_numpy(
        X_test_phi).float(), torch.from_numpy(y_test).float())

    t0 = time.process_time()

    # ADD YOUR TRAINING CODE HERE

    model = PUFWeightsPredictor(n, 1, weight_init)
    trainer = Trainer(
        model=model,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=training_size,
        train_ds=train_dataset,
        val_ds=val_dataset
    )
    trainer.train()

    t1 = time.process_time()
    training_time = t1 - t0  # time taken to get w0
    print("Training time:", training_time)
    print("Training size:", training_size)

    w0 = model.get_weights()

    # Evaluate your result
    n_test = 10000
    correct = 0
    for i in range(1, n_test+1):
        # a random challenge vector
        c_test = np.random.randint(0, 2, size=(1, n))
        r = puf_query(c_test, w)
        r0 = puf_query(c_test, w0)
        correct += (r == r0)

    success_rate = correct/n_test
    print("Success rate:", success_rate)

    # If the success rate is less than 99%, a penalty time will be added
    # One second is add for each 0.01% below 99%.
    effective_training_time = training_time
    if success_rate < 0.99:
        effective_training_time = training_time + 10000*(0.99-success_rate)
    print("Effective training time:", effective_training_time)
    results.append({
        "lr": lr,
        "num_epochs": num_epochs,
        "weight_init": weight_init,
        "training_size": training_size,
        "training_time": training_time,
        "effective_training_time": effective_training_time[0] if isinstance(effective_training_time, np.ndarray) else effective_training_time,
        "success_rate": success_rate[0]
    })
    return results


params = {
    "lr": 250,
    "num_epochs": 10,
    "weight_init": 1,
    "training_size": 10000,
}

train_and_eval(params)
