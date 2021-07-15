## REU
In this project, we are working with RGB images and hyperspectral images. The goal is to classify these images into several classes depending on the images. We use a graphical framework to classify these images because it allows for a general incorporation of information of any kind of dataset (video, text, images, etc). It is also able to work with nonlinearly separable classes. Lastly, it can capture texture and patterns throughout the image using the non-local means feature vector, which we will use in our framework.

The first step of this framework is to compute the non-local means feature vector given an image and a window size. The algorithm is specified in [1] as Algorithm 4. Then, we would create a weight matrix using a k-nearest neighbor graph. We used the graphlearning package proposed by Dr. Calder to construct the weight matrix. Next, we can perform spectral clustering which is an unsupervised learning algorithm to classify the weight matrix into k classes, which will serve as our image segmentation. Finally, we can plot the image, the segmentation, as well as the eigenvectors.

[1] Z. Meng, E. Merkurjev, A. Koniges, and A. L. Bertozzi, “Hyperspectral
image classification using graph clustering methods,” Image Processing
On Line, vol. 7, pp. 218–245, 2017.
