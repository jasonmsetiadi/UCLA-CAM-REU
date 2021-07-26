## Import useful packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.ndimage import gaussian_filter
import graphlearning as gl
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearn_pca
import urllib.request
import scipy.io

## Function for Non-Local Means Method
def NonLocalMeans(image, d):
  
  ## Pad the image with mirror reflections of itself with a width d
  pad = (d,d)
  padimage = np.pad(image,(pad,pad,(0,0)),mode='reflect') #(top,bottom),(left,right),(0,0)

  ## For the ith pixel, make a (2d + 1) by (2d + 1) patch centered at pixel i
  patches = extract_patches_2d(padimage, tuple((2*d+1,2*d+1)))

  ## For the jth, (j = 1; 2; 3) band, apply a Gaussian kernel on this patch
  u = np.zeros((2*d+1,2*d+1))
  u[d,d] = 1
  G = gaussian_filter(u,d/2,mode='constant',cval=0)
  patches = patches * G[np.newaxis,:,:,np.newaxis]

  ## Form the feature matrix F by letting each row of F be a feature vector of a pixel
  F = patches.reshape((patches.shape[0],patches.shape[1]*patches.shape[2]*patches.shape[3]))
  print("feature vector shape: ", F.shape)

  return F

#Function to perform PCA on image
def PCA(image, component=False):

  #Perform PCA with 60 components
  pca = sklearn_pca(n_components=60)
  X = image.reshape(image.shape[0]*image.shape[1],image.shape[2])
  pca.fit(X)
  variance = 100*(pca.explained_variance_ratio_)

  #Get the number of components with variance greater than 0.005%
  num_components = len(variance[variance>5e-3])

  #Perform PCA with the new number of components
  pca = sklearn_pca(n_components=num_components)
  pca_image = pca.fit_transform(X)
  print("Total Variation (%d components): "%num_components, np.sum(pca.explained_variance_ratio_))
  pca_image = pca_image.reshape(image.shape[0], image.shape[1], num_components)
  print("pca image shape: ", pca_image.shape)
  
  if component==True:
    return pca_image, num_components
  return pca_image

## Function to generate the weight matrix
def WeightMatrix(data):
  boolean = False
  k=1
  I,J,D = gl.knnsearch_annoy(data, 50, similarity='angular')
  D[D==0] = 1e-150

  while boolean==False:
    k+=1
    W = gl.weight_matrix(I,J,D,k)
    boolean = gl.isconnected(W)
    print("k=%d"%k, "connected=%s"%boolean)

  k=2*k
  W = gl.weight_matrix(I,J,D,k)
  print("k=%d"%k, "connected=%s"%boolean)
  print("weight matrix shape: ", W.shape)
  
  return W

## Function to perform spectral clustering and plot the image segmentation
def spectral_clustering(W, image, labels_rescaled, method, plot=True):

  ## Spectral Embedding
  num_classes = len(np.unique(labels_rescaled))
  L = gl.graph_laplacian(W, norm='normalized')
  vals, vec = sparse.linalg.eigs(L,k=num_classes+1,which='SM')
  vals = vals[1:].real
  vec = vec[:,1:].real  
  norm = np.sum(vec**2,axis=1)
  Y = vec/norm[:,np.newaxis]**.5 

  ## K-means
  kmeans = KMeans(n_clusters=num_classes).fit(Y)
  labels_cluster = kmeans.labels_

  if plot==True:

    ## Plot image segmentation
    image_seg = np.reshape(labels_cluster,image.shape[0:2])
    plt.figure()
    plt.title('Spectral Clustering Segmentation')
    plt.imshow(image_seg)
    plt.savefig('Spectral_Clustering_%d.png'%method)
    plt.show()

    # ## Plot eigenvectors
    # for i in range(num_classes):
    #   eigen_vec = vec[:,i].reshape((image.shape[0],image.shape[1]))
    #   plt.figure()
    #   plt.title('eigenvector %s'%i)
    #   plt.imshow(eigen_vec, cmap='gray')

  ## Accuracy score
  acc = gl.clustering_accuracy(labels_cluster, labels_rescaled)
  print("Spectral clustering accuracy: %.2f%%"%acc)

  return acc

## Function to perform laplace and poisson learning and plot the image segmentation
def laplace_poisson(W, train_ind, image, labels, method, plot=True):
  #Laplace Learning
  labels_laplace = gl.graph_ssl(W, train_ind, labels[train_ind], algorithm='laplace')
  laplace_acc = gl.accuracy(labels, labels_laplace, len(train_ind))

  if plot==True:

    ## Plot laplace segmentation
    laplace_seg = np.reshape(labels_laplace,image.shape[0:2])
    plt.figure()
    plt.title('Laplace Segmentation')
    plt.imshow(laplace_seg)
    plt.savefig('Laplace_Learning_%d.png'%method)
    plt.show()
  print('Laplace learning accuracy: %.2f%%'%laplace_acc)

  #Poisson Learning
  labels_poisson = gl.graph_ssl(W, train_ind, labels[train_ind], algorithm='poisson')
  poisson_acc = gl.accuracy(labels, labels_poisson, len(train_ind))

  if plot==True:

    ## Plot poisson segmentation
    poisson_seg = np.reshape(labels_poisson,image.shape[0:2])
    plt.figure()
    plt.title('Poisson Segmentation')
    plt.imshow(poisson_seg)
    plt.savefig('Poisson_Learning_%d.png'%method)
    plt.show()
  print('Poisson learning accuracy: %.2f%%'%poisson_acc)

  return laplace_acc, poisson_acc

## Import the hyperspectral dataset
urllib.request.urlretrieve('http://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat', 'salinas.mat')
data = scipy.io.loadmat('salinas.mat')
image = data['salinas']
num_pixels = image.shape[0]*image.shape[1]
print(image.shape)

## Display a random sample band of the image
rand_slice = np.random.randint(image.shape[2])
plt.title('salinas image (band: %d)'%rand_slice)
plt.imshow(image[:,:,rand_slice])
plt.savefig('Salinas_random_band.png')
plt.show()

## Import the ground truth labels
urllib.request.urlretrieve('http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat', 'salinas_gt.mat')
data = scipy.io.loadmat('salinas_gt.mat')
gt = data['salinas_gt']
labels = gt.flatten().astype(int)
plt.title('ground truth labels')
plt.imshow(gt)
plt.savefig('Ground_Truth.png')
plt.show()

"""#Comparing different preprocessing methods"""

#Initialize training points
np.random.seed(0)
num_train_per_class = int(0.10*num_pixels/len(np.unique(labels)))
train_ind = gl.randomize_labels(labels,num_train_per_class)
print("%.3f%% training data"%(100*len(train_ind)/num_pixels))

#Array to store accuracy scores
spectral = np.ones(5)
laplace = np.ones(5)
poisson = np.ones(5)

"""1. Raw Image"""

#Construct Weight Matrix  
raw_image = image.reshape(num_pixels, image.shape[2])
W = WeightMatrix(raw_image)

#Perform Spectral Clustering to segment the image
spectral[0] = spectral_clustering(W, image, labels, 1)

#Perform Laplace and Poisson Learning to segment the image
laplace[0], poisson[0] = laplace_poisson(W, train_ind, image, labels, 1)

"""2. Raw NonLocalMeans"""

#Perform NonLocalMeans and Construct Weight Matrix
F = NonLocalMeans(image, 2)
W = WeightMatrix(F)

#Perform Spectral Clustering to segment the image
spectral[1] = spectral_clustering(W, image, labels, 2)

#Perform Laplace and Poisson Learning to segment the image
laplace[1], poisson[1] = laplace_poisson(W, train_ind, image, labels, 2)

"""3. Raw PCA"""

#Perform PCA and Construct Weight Matrix
pca_image, raw_pca_components = PCA(image, component=True)
W = WeightMatrix(pca_image.reshape(num_pixels, pca_image.shape[2]))

#Perform Spectral Clustering to segment the image
spectral[2] = spectral_clustering(W, image, labels, 3)

#Perform Laplace and Poisson Learning to segment the image
laplace[2], poisson[2] = laplace_poisson(W, train_ind, image, labels, 3)

"""4. NonLocalMeans-PCA"""

#Perform NonLocalMeans then PCA and Construct Weight Matrix
feat_vec = NonLocalMeans(image, 2)
feat_vec = feat_vec.reshape(image.shape[0], image.shape[1], feat_vec.shape[1])
pca_image = PCA(feat_vec)
W = WeightMatrix(pca_image.reshape(num_pixels, pca_image.shape[2]))

#Perform Spectral Clustering to segment the image
spectral[3] = spectral_clustering(W, image, labels, 4)

#Perform Laplace and Poisson Learning to segment the image
laplace[3], poisson[3] = laplace_poisson(W, train_ind, image, labels, 4)

"""5. PCA-NonLocalMeans"""

#Perform PCA then NonLocalMeans and Construct Weight Matrix
pca_image = PCA(image)
F = NonLocalMeans(pca_image, 2)
W = WeightMatrix(F)

#Perform Spectral Clustering to segment the image
spectral[4] = spectral_clustering(W, image, labels, 5)

#Perform Laplace and Poisson Learning to segment the image
laplace[4], poisson[4] = laplace_poisson(W, train_ind, image, labels, 5)

#Output best accuracy for each method
spectral_ind = np.argmax(spectral)
print("Best Spectral Accuracy (method: %d): %.2f"%(spectral_ind+1, spectral[spectral_ind]))
laplace_ind = np.argmax(laplace)
print("Best Laplace Accuracy (method: %d): %.2f"%(laplace_ind+1, laplace[laplace_ind]))
poisson_ind = np.argmax(poisson)
print("Best Poisson Accuracy (method: %d): %.2f"%(poisson_ind+1, poisson[poisson_ind]))

"""#PCA-NonLocalMeans-VAE"""

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(25*raw_pca_components, 100)
        self.fc21 = nn.Linear(100, 2)
        self.fc22 = nn.Linear(100, 2)
        self.fc3 = nn.Linear(2, 100)
        self.fc4 = nn.Linear(100, 25*raw_pca_components)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 25*raw_pca_components))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 25*raw_pca_components), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data_loader.dataset)))

no_cuda = False
batch_size = 128
log_interval = 10    #how many batches to wait before logging training status
epochs = 100

#GPU settings
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

#Load SalinasA data
pca_image = PCA(image)
feat_vec = NonLocalMeans(pca_image, 2)
data = (feat_vec-feat_vec.min())/feat_vec.max()
target = labels

#Convert to torch dataloaders
data = torch.from_numpy(data).float()
target = torch.from_numpy(target).long()
salinasA = MyDataset(data, target) 
data_loader = DataLoader(salinasA, batch_size = batch_size, shuffle = True, **kwargs)

#Put model on GPU and set up optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

#Training epochs
for epoch in range(1, epochs + 1):
    train(epoch)
    scheduler.step()

#Encode the dataset and save to npz file
with torch.no_grad():
    mu, logvar = model.encode(data.to(device).view(-1, 25*raw_pca_components))
    Y = mu.cpu().numpy()
    np.savez_compressed('SalinasA_vae.npz',data=Y,labels=target.numpy())

# M = np.load('SalinasA_vae.npz')
# data = M['data']
# print(data.shape)

# W_vae = WeightMatrix(data)
# accuracy(W_vae, labels)

"""Visualization of 2 Dimensions"""

#PCA
pca = sklearn_pca(n_components=2)
pca_image = pca.fit_transform(image.reshape(num_pixels,image.shape[2]))
print(pca_image.shape)
plt.title("pca")
plt.scatter(pca_image[:,0], pca_image[:,1], c=labels)
plt.savefig('PCA_2dim.png')
plt.show()

#VAE
M = np.load('SalinasA_vae.npz')
data = M['data']
print(data.shape)
plt.title("vae")
plt.scatter(data[:,0], data[:,1], c=labels)
plt.savefig('VAE_2dim.png')
plt.show()

