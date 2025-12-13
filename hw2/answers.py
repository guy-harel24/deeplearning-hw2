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
    wstd = 0
    lr_vanilla = 0.01
    lr_momentum = 0.01
    lr_rmsprop = 0.0001
    reg = 0.0001
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
    wstd = 0.001
    lr = 0.001
    
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. Yes we got results that we expected. We planned for an overfit when the dropout is 0 and we certainly got one.
As we can see the model reaches 94% on the training set but works very poorly on the test set. That is a clear sign of overfitting.
When we increase dropout we don't get the same astounding results on the training set but we do see an improvement on the test set.
We achieved what we wanted which is improved generalization.

2. If we look at the low dropout class we see amazing accuracy and loss on the train set, but pretty consistent low accuracy and high loss
on the test set. We can assume that this occurs because the model overfitted and basically memorized the train set
which means it didn't really learn a lot about the world. With each epoch it gets better at memorizing the train set
but as we can see this doesn't translate to test set predictions.
In comparison, for high dropout, we intentionally harm the model's memorization abilities by dropping a very large number
of different neurons in each layer. This doesn't allow the model to predict well on the train set but it harms the overall learning
abilities of the model. We can see that all of the graphs are pretty constant, our model can't learn when we switch off neurons
in a very intense manner.

"""
part2_q2 = r"""
**Your answer:**

Yes it is possible that the cross entropy loss will decrease while the accuracy decreases as well.
Let's look at an example where I have one sample and I was correct by giving it the highest probability out of 10 classes $p = 0.2$
Then the value of my loss will be $L = -log(0.2) \approx 0.69$ and my accuracy will be $acc = 100\%$.
In the next iteration I could predict a wrong class with a probablilty $p = 0.4$ but give the right class a
probability of $p = 0.3$. In that case my loss will be $L = -log(0.3) \approx 0.52$ and my accuracy will be $acc = 0\%$.
"""

part2_q3 = r"""
**Your answer:**

1. Differences: a. SGD has a stochastic (random) ingredient which GD doesn't have which improves generalization.
b. SGD as more memory efficient, only a batch of samples is loaded to memory and not all of the samples.
Similarities: a. They both use a uniform learn rate for all the parameters update. b. They both converge to a minima pretty slowly.
They can fluctuate with every step.

2. Yes we believe it will be helpful to incorporate momentum in GD. If I chose to use the full batch of samples instead
of mini-batches I will still experience the problem of fluctuation and long time of convergence. As I said in the previous
subsection, the problems are the same in that regard between GD and SGD. If we choose to use momentum and incorporate
decaying old gradients will shorten the time of convergence and give less importance to wild-behaving parameters.

3. A. No it is not mathematically equivalent. That is because we use the sum of losses instead of the mean.
When we use GD every sample of ours contribute its part to the overall loss: $\bar{L} = \frac{1}{N} \sum_{i=1}^{N} L_i$
There for in the first step of backpropagation for each element $x_i$ will get $dx_i = \frac{1}{N} * \frac{\partial L_i}{\partial x_i}$.
In the other implementation we get  $\bar{L} = \sum_{i=1}^{N} L_i$ Therefore each element's downstream gradient will be $\frac{\partial L_i}{\partial x_i}$.
B. We believe that the thing that went wrong is that all of the neurons layers computed by the model were save in a cache of some sort
in order to enable the backward pass later. Each layer need to remember the output it was given in the forward pass before computing the gradient.
C. We believe that we can solve it by either reverting back to SGD and let the parameters learn from each batch separately.
Another option is to accumulate gradients with the addition of scaling the batch loss. If we have N batches we will the scale the batch loss by $\frac{1}{N}$
before backpropagation then we accumulate the gradients in the computational graph that way we need to save always one
gradient for each layer. we just update its value for each batch before we update the parameters.

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