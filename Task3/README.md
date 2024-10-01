Project 3: Image Similarity

We needed to do a classification on images based no similarity. The idea was
that for a image triple (A,B,C) we classify as 1 if A is more similar to B, than B to C. And 0 otherwise.

We had to proceed as follows: <br>
1. First we needed to enconde the image to tensor, in order to be able to feed them to our neural network.
2. We used Transfer Learning, i.e.: we encoded our images using already trained very large neural networks, which we downloaded from the PyTorch Library.
3. Our encoded image triple became vectors, which we in turn stacked to create our data matrix.
4. We then created a simple neural network and trained it on our new data matrix.