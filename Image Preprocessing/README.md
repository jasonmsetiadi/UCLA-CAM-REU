## Image Preprocessing
In this section, I explored image preprocessing methods for RGB and Hyperspectral image datasets. To compare the performance of these methods, I performed both unsupervised learning and semi-supervised learning to classify those images. I use Spectral Clustering [3] as the unsupervised learning method while Laplace Learning [4] and Poisson Learning [5] as the semi-supervised learning methods.

I performed segmentation on the raw image as the baseline result to be compared with the other methods. The preprocessing methods I want to try out are Non-local means [1], Principal Component Analysis, and Variational Autoencoder [6]. Non-local means tries to create a square patch surrounding each pixel to help improve the classifier's performance. Non-local means is great for capturing texture and patterns throughout the image while PCA and VAE are both dimentionality reduction tools. I also tried to combine those methods to see how they perform, such as Non-local means with PCA, Non-local means with VAE, as well as combining the three of them together.

After preprocessing the image, I created a weight matrix using a k-nearest neighbor graph. I used the graphlearning package [2] proposed by Dr. Calder to construct the weight matrix. Then, I can perform the unsupervised and semi-supervised learning methods and plot the image segmentation as well as the accuracy scores if possible. Lastly, I plotted the 2d result comparing PCA with 2 components and PCA with 10 components then non-local means with window size 2 then reduce the dimension to 2 using the variational autoencoder.

Based on the results, what I concluded was that PCA and Non-local means are great preprocessing methods for hyperspectral images. It helps to improve the accuracy in both unsupervised and semi-supervised learning as well as produce a well constructed graph. The VAE didn't perform as well as PCA and Non-local means for these hyperspectral datasets. However, VAE might perform well for other hyperspectral image datasets.

## Notes
1. The TwoCows RGB image serves as our initial exploration on this image preprocessing field since it is relatively small and has a clear segmentation.

2. Then I explored SalinasA which is a subset of the Salinas Hyperspectral dataset. I performed the complete exploration of the preprocessing methods specified on this dataset. 

3. Finally I also ran all the preprocessing methods on the whole Salinas Hyperspectral dataset through Dr. Calder's remote server since the dataset is relatively large.

## References

[1] Z. Meng, E. Merkurjev, A. Koniges, and A. L. Bertozzi, “Hyperspectral
image classification using graph clustering methods,” Image Processing
On Line, vol. 7, pp. 218–245, 2017.

[2] jwcalder, GraphLearning, (2020), GitHub repository, https://github.com/jwcalder/GraphLearning

[3] A. Y. Ng, M. I. Jordan, and Y. Weiss. On spectral clustering: Analysis and an algorithm. In NIPS, pages 849–856, 2001.

[4] X. Zhu, Z. Ghahramani, and J. Lafferty, “Semi-supervised learning
using Gaussian fields and harmonic functions,” in Proceedings of the
TIntieth International Conference on International Conference on
Machine Learning, ser. ICML’03. AAAI Press, 2003, p. 912–919.

[5] J. Calder, B. Cook, M. Thorpe, and D. Slepˇcev, “Poisson learning:
Graph based semi-supervised learning at very low label rates,” in 37th
International Conference on Machine Learning, ICML 2020, ser. 37th
International Conference on Machine Learning, ICML 2020, H. Daume
and A. Singh, Eds. International Machine Learning Society (IMLS),
2020, pp. 1283–1293.

[6] Z. Cao, X. Li, and L. Zhao. Unsupervised feature learning by autoencoder and prototypical contrastive learningfor hyperspectral classification.arXiv preprint arXiv:2009.00953, 2020.
