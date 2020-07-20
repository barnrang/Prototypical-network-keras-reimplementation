# Reimplementation of Prototypical Network using Keras (TF 2.0)
Blog post: https://medium.com/@barnrang/re-implementation-of-the-prototypical-network-for-few-shot-learning-using-tensorflow-2-0-keras-b2adac8e49e0

This repository is a reimplementation of the paper "Prototypical Networks for Few-shot Learning"

### Citing this data set
Please cite the following paper:


[Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction.](http://www.sciencemag.org/content/350/6266/1332.short) _Science_, 350(6266), 1332-1338.


We are grateful for the [Omniglot](http://www.omniglot.com/) encyclopedia of writing systems for helping to make this data set possible, and for [Jason Gross](https://people.csail.mit.edu/jgross/) who was essential to the development and collection of this data set.


### CONTENTS
The Omniglot data set contains 50 alphabets total. We generally split these into a background set of 30 alphabets and an evaluation set of 20 alphabets.  

To compare with the results in our paper, only the background set should be used to learn general knowledge about characters (e.g., hyperparameter inference or feature learning). One-shot learning results are reported using alphabets from the evaluation set.

A more challenging representation learning task uses the smaller background sets "background small 1" and "background small 2". Each of these contains just 5 alphabets, more similar to the experience that a human adult might have in learning about characters in general.  Our paper reports a large set of results on the 30 background alphabets, as well as results for several models on these smaller, more challenging background sets.

### Reference
[1] Jake Snell and Kevin Swersky and Richard S. Zemel (2017). Prototypical Networks for Few-shot LearningCoRR, abs/1703.05175.
