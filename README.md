## REU
In this project, we are working on image segmentation as well as image classification problems. The main difference is, in image segmentation, each data point is a pixel of an image while in image classification, each data point is a single image. In our experiments, we use RGB and Hyperspectral images for tackling image segmentation while for image classification, we use datasets like MNIST and MSTAR. We use a graphical framework in our experiments because it allows for a general incorporation of information of any kind of dataset (video, text, images, etc). It also enables us to work with nonlinearly separable datasets. We used Spectral Clustering[3] for our unsupervised learning framework while Laplace[4] and Poisson Learning[5] for our semi-supervised learning frameworks.

In our experiments of image segmentation, we explored image preprocessing methods such as Non-local means[1], PCA, and Variational Autoencoders[6] to see how they compare with the raw image segmentation. We discovered that those preprocessing methods actually help improve the accuracy of the segmentation compared to segmenting the raw image. In my opinion, the best approach would be to perform PCA on the raw image with a considerable amount of components to achieve high variation, then perform Non-local means to help improve the classifier's performance. The reason is that this works well especially with large images because PCA helps reduce the dimension and Non-local means helps the classifier classify each pixel better, resulting in less computation time while still improving the image segmentation accuracy.

We also explored the use of active learning methods on a semi-supervised learning framework to see if they perform better than random sampling. We discovered that active learning is indeed much better than random sampling and is worth trying on semi-supervised learning frameworks. We tried out uncertainty sampling, v-optimality[7], sigma-optimality[8], model change, as well as model change - voptimality. Based on our results, seems like v-optimality and model change - voptimality are the best performing ones, especially with low label rates. This is useful in general as labeling is expensive and time consuming.

This project is mentored by Dr. Jeff Calder (U of Minnesota Math Department), Dr. Andrea Bertozzi (UCLA Math Department), and Kevin Miller (UCLA Math Department).

## References

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

[7] M. Ji and J. Han, “A variance minimization criterion to active learning on graphs,” in Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, N. D. Lawrence and M. Girolami, Eds., vol. 22. La Palma, Canary Islands: PMLR, 21–23 Apr 2012, pp. 556–564. [Online]. Available: http://proceedings.mlr.press/v22/ji12.html

[8] Y. Ma, R. Garnett, and J. Schneider, “Σ-optimality for active learn- ing on Gaussian random fields,” in Advances in Neural Information Processing Systems, vol. 26, 2013.
