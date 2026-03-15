## Word2Vec CBOW (NumPy Implementation)

A minimal implementation of the **Word2Vec Continuous Bag of Words (CBOW)** model written in pure **NumPy**, trained on *White Fang* by Jack London.

The project demonstrates the full training pipeline of Word2Vec without relying on deep learning frameworks.

### Features

- CBOW architecture implemented from scratch
- Negative sampling for efficient training
- Pure NumPy implementation (no PyTorch / TensorFlow)
- Full optimization loop:
  - forward pass
  - loss computation
  - gradient calculation
  - parameter updates
- Training progress reporting:
  - accumulated loss
  - average loss
- Basic similarity checks between learned word embeddings
- Export of trained embeddings to a text file

### Dataset

The model is trained on the novel **White Fang** by Jack London.

### Output

After training, the learned word embeddings are saved to a text file and can be used for:

- word similarity
- clustering
- downstream NLP experiments
