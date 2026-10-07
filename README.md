# epoch

a swift playground for learning how neural networks actually learn.

you start with basically nothing.

draw a number. the model guesses. you see how wrong it was. then you change the weights and try again.

that's pretty much the whole idea.

## what you'll learn

epoch introduces the maths as you go:

* weights and biases
* activations
* probabilities
* loss
* gradients
* gradient descent
* training

you don't need to know any ai beforehand.

## mnist

the model learns to recognise handwritten digits from mnist.

the playground uses a small subset of the dataset so everything can stay inside the playground's size limit.

you can also draw your own digits and see what the model thinks.

## implementation

built in swift.

the neural network and training loop are implemented directly so you can actually see what's going on instead of calling a training library and getting a finished model back.

## status

wip.

the first goal is just to make neural-network training make sense.
