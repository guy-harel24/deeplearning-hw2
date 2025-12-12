r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0
    lr = 0.02
    reg = 0.1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3  # Depth: 3 hidden layers is usually a good balance to start.
    hidden_dims = 256  # Width: Powers of 2 are standard; 256 provides decent capacity.
    activation = "relu"  # Standard choice, avoids vanishing gradients better than sigmoid/tanh.
    out_activation = "none"  # Important: PyTorch CrossEntropyLoss expects raw logits, not probabilities.
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss() 
    lr = 0.01
    weight_decay = 1e-4
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

**Question 1**
1. **Optimization error**: arises from limitations of the training algorithm, which represents the gap between the hypothesis model from hypothesis class, and the best possible hypothesis to choose from it. This error is due to choosing poor optimization hyperparameters, or since the optimization algorithm did not manage to find the best hypothesis due to local minimum for example, where it got stuck at. This error can be reduced by using a better optimizer for the data.
2. **Generalization error**: arises from difference between the performence over the training data and evaluation data. This error is a classis outcome of overfitting. This error can be reduced by adding regularization, or adding more data to the training batch.
3. **Approximation error**: arises from limitations of the hypothesis class, where even its best hypothesis does not ensure high accuracy. This could be explained as a gap between the "true" hypothesis matching the data, to the best in the hypothesis class. This error can be reduced by enhancing the hypothesis class.

**Question 2**
1. **Optimization error**: the model manages to get high accuracy over the training data, which means the optimizer itself did its job, resulting in **low** optimization error.
2. **Generalization error**: although the loss over the training data is low, the high loss over the evaluation data shows high overfitting, meaning that the egneralization error is **high**.
3. **Approximation error**: since the accuracy over the model's training data is high, it's very likely that with additional data the hypothesis class would have been sufficient. We can conclude that the approximation error is **low**


"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 1e-4  # L2 Regularization to prevent overfitting
    momentum = 0.9       # Accelerates SGD in the relevant direction
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part4_q2 = r"""
**Question 1**

First, let's calculate The Jacobian matrix $\frac{\partial y_1}{\partial x_1}$:
    
$$\forall i \le m, j \le n \quad \left[ \frac{\partial y_1}{\partial x_1} \right]_{i, j} = \frac{\partial y_{1, i}}{\partial x_{1, j}}$$
    
Using the definition above we get $y_{1, i} = \sum_{k=1}^{n} M_{i k} x_{1, k}$:
    
$$\frac{\partial y_{1, i}}{\partial x_{1, j}} = \frac{\partial}{\partial x_{1, j}} \left( \sum_{k=1}^{n} M_{i k} x_{1, k} \right) = \frac{\partial}{\partial x_{1, j}} (M_{i, j} x_{1, j}) = M_{i j} \Rightarrow \frac{\partial y_1}{\partial x_1} = M$$

    
We can now apply the chain rule:
    
$$\frac{\partial L}{\partial x_1} = \left(\frac{\partial y_1}{\partial x_1}\right)^T \cdot \frac{\partial L}{\partial y_1} \Rightarrow \frac{\partial L}{\partial x_1} = M^T \frac{\partial L}{\partial y_1}$$


**Question 2**

We calculate the Jacobian $\frac{\partial y_2}{\partial x_2}$ using the sum rule for derivatives:

$$\frac{\partial y_2}{\partial x_2} = \frac{\partial (x_2)}{\partial x_2} + \frac{\partial (M x_2)}{\partial x_2} = I + M$$

Applying the chain rule:
$$\frac{\partial L}{\partial x_2} = \left( \frac{\partial y_2}{\partial x_2} \right)^T \frac{\partial L}{\partial y_2} \Rightarrow \frac{\partial L}{\partial x_2} = (I + M)^T \frac{\partial L}{\partial y_2} \Rightarrow \frac{\partial L}{\partial x_2} = (I + M^T) \frac{\partial L}{\partial y_2}$$

**Question 3**

We assumed that $M$ has small entries, which is standard for weight initialization.

When multiplying many matrices with small norms together, the product might approache zero exponentially fast.
In the standard case, the gradient signal passes exclusively through the weights $M$. 

Since $M$ is small, the gradient effect gets smaller at every single step.
After dozens of layers, the "signal" is effectively (or literally) destroyed, and the early layers stop learning.

In the residual case, the gradient is multiplied by $(I + M^T)$. This creates two "pathways" for the gradient:
1. The Residual Path ($M^T$): The standard learned path, which might shrink the gradient.
2. The Identity Path ($I$): The term $\frac{\partial L}{\partial y_2}$ is passed directly to the previous layer unchanged (multiplied by 1).

Even if $M$ is very small (or even zero due to multiplications results and/or numerical errors), the gradient can flow backward through the Identity term without being diminished. This ensures that the gradient signal can reach the earliest layers of the network regardless of how deep the network is, effectively solving the vanishing gradient problem.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

**Question 1**

The model detected pretty poorly. It did not detect all objects correctly (and even missed the cat), and it miss-classified all of them (except one dog with low confidance). Some false classifications came with pretty high confidance levels (one dog as a cat, and one dolphin as a person.

**Question 2**


"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""