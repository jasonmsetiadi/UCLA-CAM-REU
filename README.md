## REU
In this project, we are working with RGB images and hyperspectral images. The goal is to classify these images into several classes depending on the images. We use a graphical framework to classify these images because it allows for a general incorporation of information of any kind of dataset (video, text, images, etc). It is also able to work with nonlinearly separable classes. Lastly, it can capture texture and patterns throughout the image using the non-local means feature vector, which we will use in our framework. The algorithm is specified in [1] as Algorithm 4.

We will try out several image preprocessing methods and compare the results from spectral clustering [3], laplace learning [4], and poisson learning [5]. The methods are non-local means, PCA, non-local means then PCA, as well as PCA then non-local means. We also have a baseline result using the raw image.

After preprocessing the image, we can create a weight matrix using a k-nearest neighbor graph. We used the graphlearning package [2] proposed by Dr. Calder to construct the weight matrix. Then, we can perform spectral clustering which is an unsupervised learning algorithm, laplace and poisson learning which are both semi-supervised learning algorithms, to classify the image into k classes, which will serve as our image segmentation. Finally, we can plot the image segmentation as well as the accuracy scores if possible.

Additionally, we also explored the use of variational autoencoders [6] which serves as a dimensionality reduction tool. We plotted the 2d result comparing PCA with 2 components and PCA with 10 components then non-local means with window size 2 then reduce the dimension to 2 using the variational autoencoder.

[1] Z. Meng, E. Merkurjev, A. Koniges, and A. L. Bertozzi, “Hyperspectral
image classification using graph clustering methods,” Image Processing
On Line, vol. 7, pp. 218–245, 2017.

[2] jwcalder, GraphLearning, (2020), GitHub repository, https://github.com/jwcalder/GraphLearning

[3] A. Y. Ng, M. I. Jordan, and Y. Weiss. On spectral clustering: Analysis and an algorithm. In NIPS, pages 849–856, 2001.

[4] X. Zhu, Z. Ghahramani, and J. Lafferty, “Semi-supervised learning
using Gaussian fields and harmonic functions,” in Proceedings of the
Twentieth International Conference on International Conference on
Machine Learning, ser. ICML’03. AAAI Press, 2003, p. 912–919.

[5] J. Calder, B. Cook, M. Thorpe, and D. Slepˇcev, “Poisson learning:
Graph based semi-supervised learning at very low label rates,” in 37th
International Conference on Machine Learning, ICML 2020, ser. 37th
International Conference on Machine Learning, ICML 2020, H. Daume
and A. Singh, Eds. International Machine Learning Society (IMLS),
2020, pp. 1283–1293.

[6] Z. Cao, X. Li, and L. Zhao. Unsupervised feature learning by autoencoder and prototypical contrastive learningfor hyperspectral classification.arXiv preprint arXiv:2009.00953, 2020.
