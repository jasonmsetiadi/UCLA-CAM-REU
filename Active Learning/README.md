## REU
In this, I explored active learning methods along graph-based semi-supervised learning frameworks on datasets such as MNIST, MSTAR, as well as SalinasA Hyperspectral Image. We are interested to see whether they outperform random sampling on our experiments. There are generally two tasks in active learning, exploration and exploitation. Exploration methods are methods that explore the structure of the dataset first while exploitation methods select points that are most ambiguous for the classifier. The methods we tried out are: uncertainty sampling, v-optimality[1], sigma-optimality[6], model change, and model change - voptimality. Uncertainty is an example of exploitation methods while v and sigma opt are well-known exploration methods. Model change and model change - vopt are methods that try to balance out exploration with exploitation with different ratios.

We use two GSSL frameworks which are Laplace and Poisson learning in our experiments. First, we choose an initial set of labeled points from each class randomly using a function from Dr Calder's package[2]. Then, we start applying our methods to select the next labeled points and perform Laplace and Poisson learning to predict the remaining labels and calculate the accuracy of the prediction. Our results show that most of our active learning methods outperform random sampling, which is what we are looking for. However, in my opinion, the best performing methods would be vopt and mc-vopt because they have a significant increase with low label rates but also explored ambiguous regions pretty well as labeled points increases.

At first, we tested our active learning methods on a synthetic dataset with 8 gaussian clusters. This enables us to visualize how each method selects the points to label. One drawback we had was with model change because it didn't seem to perform as we expected. However, the other methods perform as expected and so we can apply those methods to more complicated datasets as listed in the beginning. One guess as to why model change is not performing as expected may because the covariance matrix still contains negative entries which isn't supposed to happen.

## Notes
1. To view Toy_Dataset_Active_Learning.ipynb, go to https://nbviewer.jupyter.org/ and paste https://github.com/jasonmsetiadi/UCLA-CAM-REU/blob/main/Active%20Learning/Toy_Dataset_Active_Learning.ipynb to view the notebook.

## References
	
[1] M. Ji and J. Han, “A variance minimization criterion to active learning on graphs,” in Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, N. D. Lawrence and M. Girolami, Eds., vol. 22. La Palma, Canary Islands: PMLR, 21–23 Apr 2012, pp. 556–564. [Online]. Available: http://proceedings.mlr.press/v22/ji12.html

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

[6] Y. Ma, R. Garnett, and J. Schneider, “Σ-optimality for active learn- ing on Gaussian random fields,” in Advances in Neural Information Processing Systems, vol. 26, 2013.
