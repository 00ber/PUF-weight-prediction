import time
import scipy
import numpy as np
from sklearn.model_selection import train_test_split

# Custom Binary Cross Entropy Loss


class BCELossValue:
    def __init__(self, y_pred, y_true, value):
        self.y_pred = y_pred
        self.y_true = y_true
        self.value = value

    def backward(self, epsilon=1e-12, next_grad=1):
        z = np.concatenate([self.y_true, self.y_pred], axis=1)

        def calc_loss(y_row):
            y_tr, y_pr = y_row
            if y_tr == 1:
                return -1/(y_pr + epsilon)
            elif y_tr == 0:
                return 1/(1 - y_pr + epsilon)
            else:
                raise Exception("Invalid y_pred value")

        val = np.apply_along_axis(calc_loss, 1, z)
        val = val.reshape(-1, 1)
        return val

# Custom Sigmoid activation


class SigmoidValue:
    def __init__(self, x, value):
        self.x = x
        self.value = value

    def backward(self, loss_grad):
        return loss_grad * (self.value * (1 - self.value))

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __truediv__(self, other):
        return self.value / other

    def ___rtruediv__(self, other):
        return other / self.value


def sigmoid_fn(x):
    # value = 1/(1 + np.exp(-1 * x))
    # Using scipy for sigmoid because np.exp is unstable for larger exponents
    value = scipy.special.expit(x)
    return SigmoidValue(x, value)


def bce_loss(y_pred, y_true, epsilon=1e-12):
    value = -y_true * np.log(y_pred.value + epsilon) - \
        (1 - y_true) * np.log(1 - y_pred.value + epsilon)
    return BCELossValue(y_pred.value, y_true, value)


class PUF:
    def __init__(self, n, low=-10, high=10):
        self.n = n
        self.weight = np.random.uniform(low=low, high=high, size=(n + 1, 1))
        self._backward = lambda: None

    def __call__(self, phis):
        out = phis @ self.weight
        return sigmoid_fn(out)

    def _calculate_gradients(self, phis, activation_grad):
        return activation_grad * phis

    def backward(self, phi, logits, loss, learning_rate):
        loss_gradient = loss.backward()
        activation_gradient = logits.backward(loss_gradient)
        batch_gradient = self._calculate_gradients(phi, activation_gradient)
        avg_gradient = np.mean(
            batch_gradient, axis=0).reshape(self.weight.shape)
        self.update(avg_gradient, learning_rate)

    def parameters(self):
        return self.weight

    def update(self, gradient, learning_rate):
        self.weight += -1 * learning_rate * gradient


# Function to train the model using the provided hyperparameters
def train(model, num_epochs, lr, X_train, X_test, y_train, y_test):

    for k in range(num_epochs):
        avg_training_loss = 0.0
        total_training_loss = 0.0
        avg_val_loss = 0.0
        total_val_loss = 0.0

        ypred = model(X_train)
        loss = bce_loss(ypred, y_train)
        model.backward(X_train, ypred, loss, lr)
        total_training_loss += np.mean(loss.value)
        avg_training_loss = total_training_loss

        ypred = model(X_test)
        loss = bce_loss(ypred, y_test)
        total_val_loss += np.mean(loss.value)
        avg_val_loss = total_val_loss
        avg_val_loss = total_val_loss
        print(
            f"[{k}/{num_epochs}] Avg Training Loss: {avg_training_loss} Avg Validation Loss: {avg_val_loss}")


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
            phi[i] = (2*c[0,i]-1)*phi[i+1]

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
    y = np.zeros((target_training_size, 1))

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


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train_phi = np.apply_along_axis(calc_phi, 1, X_train)
    X_test_phi = np.apply_along_axis(calc_phi, 1, X_test)
    print(f"Done!")

    n = 64

    w0 = np.zeros((n+1, 1))  # The estimated value of w.
    # Try to estimate the value of w here. This section will be timed. You are
    # allowed to use the puf_query function here too, but it will count towards
    # the training time.

    print(f"Starting training for {num_epochs} epochs.")
    print("-" * 100)
    t0 = time.process_time()
    # ADD YOUR TRAINING CODE HERE

    model = PUF(n, weight_init[0], weight_init[1])

    train(model, num_epochs, lr, X_train_phi, X_test_phi, y_train, y_test)

    t1 = time.process_time()
    training_time = t1 - t0  # time taken to get w0
    print("Training time:", training_time)
    print("Training size:", training_size)

    old_w0 = w0

    wt = model.weight
    # wt = state["fc.weight"]
    w0 = wt

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
    "lr": 400,
    "num_epochs": 15,
    "weight_init": (-5, 5),
    "training_size": 15000
}

train_and_eval(params)
