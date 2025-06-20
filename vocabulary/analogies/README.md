## Analogies

VexIR2Vec supports analogy queries of the form:
a : b :: c : ?

These analogies assess semantic relationships between instructions or operands within the learned embedding space. For example:

> `"get" is to "put" as "load" is to "store"`

To answer such queries, VexIR2Vec computes the vector closest to `b − a + c` using Euclidean distance.

We designed **90 analogy queries** spanning operators, types, arguments, and their semantics.

Notably, the embeddings capture meaningful relations, such as:
- Multiplication/division via left/right shift
- Operand-type distinctions across operations


**To run the analogy evaluations, refer to** [`./analogies.ipynb`](./analogies.ipynb).


## Clustering

V𝑙𝑜𝑜𝑘𝑢𝑝 encodes semantic entities as points in a 128-dimensional Euclidean space, where semantically similar entities are positioned close together.

To visualize this, we apply **t-SNE** to project the embeddings onto a 2D surface. Entities are grouped into **nine logical categories**, based on data types (e.g., integers, floats, vectors), operation types (e.g., loads, stores, logic ops), or whether they represent **user-defined symbols** (e.g., variables, functions, constants).

This clustering helps evaluate the **quality of the seed embedding** and its ability to encode semantic structure.

**To run the clustering analysis, refer to** [`./clustering.ipynb`](./clustering.ipynb).
