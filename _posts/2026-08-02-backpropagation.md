---
layout: post
title: "backpropagation"
description: "understanding the intuition behind backpropagation"
date: 2026-08-02 18:00:00
categories: [ml, dl]
tags: [neural nets, backpropagation, machine learning, deep learning]
featured: true
---

The basic intuition behind $$\frac{dy}{dx}$$ is to understand how $y$ changes with respect to a small change in $x$.

If you take a loss function, there can be many such $x$ values, and we want to know how the loss $y$ changes with respect to each of them. These $x$ values are at the leaf level of the computation graph. We can easily compute the immediate transformation outputs and their loss with respect to the previous input, but in a loss function there can be many transformations and logical operations happening between these $x$ values and the final output $y$. So how do we handle that?

This is where backward propagation comes in. It uses the chain rule, which can be derived from the basic derivative formula.

Let:

$$
y = t^2 + 2
$$

$$
t = 3x
$$

Then:

$$
\frac{dy}{dx} = \frac{dy}{dt} \cdot \frac{dt}{dx}
$$

So the derivative propagates backward through the computation graph. This lets us adjust the inputs, or more importantly the weights, to minimize the loss function.

##### How do ensure that we compute the backward loss in the correct order?

A key question is: if $$\frac{dy}{dx}$$ depends on many derivatives in between, how do we know the order in which to compute them? The answer is through **topological sorting**.

##### Training loop
In a loop:
1. Forward pass: calculate the loss
2. Adjust weights
3. Backward pass: update gradients

This process trains the neural network to find weights that minimize the loss.

**Some code I wrote while learning this topic (thanks to Karpathy)**: [https://github.com/khadyothan/aiml/blob/main/backpropagation.ipynb](https://github.com/khadyothan/aiml/blob/main/backpropagation.ipynb)