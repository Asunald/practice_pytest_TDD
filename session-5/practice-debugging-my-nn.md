# Python Debugging: `practice-my_nn_with_bugs.py`

## Introduction

Programming rarely works perfectly on the first attempt. Unexpected behavior, incorrect results, and runtime errors are normal parts of software development. The ability to **systematically locate and fix problems** is therefore one of the most important programming skills.

This tutorial introduces practical debugging techniques in Python using the file `practice-my_nn_with_bugs.py`. The focus is not machine learning itself, but **how to observe program behavior, inspect variables, and diagnose issues effectively**.

The debugging practices covered include:

* Print debugging
* Assertions
* Shape checking
* Python breakpoints (`pdb`)
* IDE debugging tools (VSCode / PyCharm)
* Systematic debugging workflow

---

# 1. Understanding the Program Before Debugging

Before debugging any program, it is important to understand its structure. Debugging without understanding the code often leads to confusion.

The file `practice-my_nn_with_bugs.py` contains several main components:

```
load_mnist_from_csv()
Layer base class
Dense layer
ReLU activation
softmax + cross-entropy
forward pass
training loop
```

For example, the forward pass of the network is implemented as:

```python
def forward(network, X):
    activations = []
    input = X

    for layer in network:
        input = layer.forward(input)
        activations.append(input)

    return activations
```

This function passes the input through each layer in sequence and records the intermediate activations.

Understanding where data flows through the program is essential before attempting to debug it.

---

# 2. Print Debugging

The simplest and most widely used debugging method is **printing intermediate values**.

Printing allows inspection of:

* variable values
* array shapes
* program execution flow

The `load_mnist_from_csv` function already contains useful debugging prints:

```python
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")
```

These outputs confirm whether the dataset was loaded correctly.

Example output:

```
Training data shape: (54000, 784)
Training labels shape: (54000,)
Validation data shape: (6000, 784)
Validation labels shape: (6000,)
```

If a dataset is loaded incorrectly, these prints reveal the problem immediately.

---

## Printing Intermediate Results

Print debugging can also help inspect how data moves through the network.

The forward function can be modified as follows:

```python
def forward(network, X):
    activations = []
    input = X

    for i, layer in enumerate(network):
        input = layer.forward(input)
        print(f"Layer {i} output shape:", input.shape)
        activations.append(input)

    return activations
```

Example output:

```
Layer 0 output shape: (54000, 64)
Layer 1 output shape: (54000, 64)
Layer 2 output shape: (54000, 32)
Layer 3 output shape: (54000, 32)
Layer 4 output shape: (54000, 10)
```

If an unexpected shape appears, the location of the problem becomes clear.

---

# 3. Assertions

Assertions are a professional technique used to enforce assumptions about program state.

An assertion verifies that a condition is true. If the condition fails, the program stops with an error.

Syntax:

```python
assert condition, "error message"
```

Assertions can be added to verify that inputs have correct dimensions.

Example inside the `Dense` layer:

```python
def forward(self, input):

    assert input.shape[1] == self.weights.shape[0], \
        "Input dimension does not match weight dimension"

    self.input = input
    return np.dot(input, self.weights) + self.biases
```

If the condition fails, Python raises an error:

```
AssertionError: Input dimension does not match weight dimension
```

Assertions are useful because they:

* detect invalid states early
* document assumptions in code
* prevent silent errors

---

# 4. Shape Checking

Many numerical programs fail because arrays have incorrect shapes.

In `practice-my_nn_with_bugs.py`, the following shapes are expected:

```
Input data: (N, 784)
First Dense layer weights: (784, 64)
Hidden layer output: (N, 64)
Second Dense layer output: (N, 32)
Final logits: (N, 10)
```

Printing shapes is often the fastest way to detect issues.

Example:

```python
print("Logits shape:", logits.shape)
```

Or enforcing shape expectations:

```python
assert logits.shape[1] == 10
```

Consistently checking shapes helps ensure that matrix operations behave as intended.

---

# 5. Python Breakpoints (`pdb`)

Python includes a built-in interactive debugger called `pdb`.

A breakpoint pauses program execution and allows inspection of variables.

To insert a breakpoint:

```python
import pdb
pdb.set_trace()
```

Example in the training function:

```python
def train(network, X, y):

    activations = forward(network, X)

    import pdb
    pdb.set_trace()

    logits = activations[-1]
```

When the program reaches this line, execution stops and the debugger opens.

Example interaction:

```
(Pdb) p logits.shape
(54000, 10)

(Pdb) p y[:10]
[5 0 4 1 9 2 1 3 1 4]
```

Useful commands:

| Command      | Description        |
| ------------ | ------------------ |
| `n`          | execute next line  |
| `s`          | step into function |
| `p variable` | print variable     |
| `c`          | continue execution |
| `q`          | quit debugger      |

The debugger allows inspection of program state at any moment.

---

# 6. IDE Debugging (VSCode / PyCharm)

Modern development environments provide visual debugging tools.

These tools allow:

* placing breakpoints with the mouse
* stepping through code line-by-line
* inspecting variables in real time
* viewing the call stack

Example workflow:

1. Place a breakpoint inside the `train()` function.
2. Run the program in debug mode.
3. Inspect variables such as:

```
logits
grad_output
weights
biases
```

Watching how these values change helps understand program behavior.

---

# 7. A Systematic Debugging Workflow

Effective debugging follows a structured approach.

### Step 1 — Reproduce the Problem

Identify the exact behavior that is incorrect.

Example:

```
Validation accuracy remains constant.
```

---

### Step 2 — Narrow Down the Source

Possible locations include:

```
data loading
forward pass
loss computation
backpropagation
parameter updates
```

---

### Step 3 — Inspect Internal State

Use debugging tools to inspect variables:

```
print()
assert
pdb
IDE debugger
```

---

### Step 4 — Verify the Fix

After fixing the problem, verify that the program behaves as expected.

Typical indicators include:

```
loss decreasing during training
reasonable output values
correct array shapes
```

---

# 8. Key Debugging Principles

Effective debugging relies on several simple practices:

Always inspect intermediate values.

Check shapes when working with numerical arrays.

Use assertions to enforce assumptions.

Use breakpoints to examine program state.

Debug one issue at a time.

---

# Conclusion

Debugging is a core part of programming. Understanding how to observe program behavior and systematically locate problems makes complex programs much easier to maintain.

The techniques introduced in this tutorial — print debugging, assertions, shape checking, and breakpoints — form a practical foundation for debugging Python programs such as `practice-my_nn_with_bugs.py`.
