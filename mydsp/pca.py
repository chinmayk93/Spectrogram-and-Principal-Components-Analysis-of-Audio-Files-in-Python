#Chinmay Kulkarni 

#Redid-820900828

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def genData(N, dim):
    """genData(N, dim) - Return N samples of dim dimensional normal data at random positions
        returns(data, mu, var)  where mu and var are the mu/var for each dimension
    """
    # Generate random parameters 
    mu = np.random.rand(dim) * 50
    mu[2] = mu[2] + 70  # Raise z mean so above projection
    var = np.array([50, 100, 5]) + np.random.rand(dim) * 10
        
    
    # Build up random variance-covariance matrix
    varcov = np.zeros([dim, dim])
    # Fill off diagonals in lower triangle
    for i in range(1, dim):
        varcov[i, 0:i] = np.random.rand(i) * 5
    # make symmetric
    varcov = varcov + np.transpose(varcov)
    # add in variances
    varcov = varcov + np.diag(var)
    
    data = np.random.multivariate_normal(mu, varcov, N)
    return (data, mu, varcov)
        
    
    

class PCA(object):
    '''
    PCA
    '''


    def __init__(self, data, corr_anal=False):
        '''
        PCA(data)
        data matrix is N x Dim
        Performs variance-covariance matrix or correlation matrix
        based principal components analysis.
        '''
        
        # subtract the mean from the data
        # We use the scipy.signal.detrend fucntion
        # Mean subtraction is a simple form of detrending.
        ddata = sig.detrend(data, axis=1, type="constant")

        if not corr_anal:
            self.Sigma = np.cov(ddata, rowvar=False)
            # Compute eigen values and vectors.
            self.eigval, self.eigvec = np.linalg.eig(self.Sigma)
            self.dim = self.Sigma.shape[0]
            self.anal_type = "variance-covariance"
        else:
            self.R = np.corrcoef(ddata, rowvar=False)  # variance-covariance matrix
            # Compute eigen values and vectors.
            self.eigval, self.eigvec = np.linalg.eig(self.R)
            self.dim = self.R.shape[0]
            self.anal_type = "autocorrelation"

        
        # Eigen vectors and values are not sorted, so rearrange them by
        # eigenvalue indices, largest value to smallest
        permute_order = np.flip(np.argsort(self.eigval), 0)
        self.eigval = self.eigval[permute_order]
        self.eigvec = self.eigvec[:, permute_order]
        
    def get_pca_directions(self):
        """get_pca_directions() - Return matrix of PCA directions
        Each column is a PCA direction, with the first column
        contributing most to the overall variance.
        """
        return self.eigvec
             
    def get_contributions_to_var(self, cumulative=False):
        """get_contributions_to_var(self, cumulative=False)
        Show the amount that each of the principal component axes contributes
        to capturing the variance.  If cumulative is True
        """
        
    def transform(self, data, dim=None):
        """transform(data, dim) - Transform data into PCA space
        To reduce the dimension of the data, specify dim as the remaining 
        number of dimensions
        """
        if not dim:
            dim = self.dim  # Use all if user didn't specify
        
        # Matrix multiplication:  data * selected eigen vectors
        return np.dot(data, self.eigvec[:, 0:dim])
        
        
    def get_component_loadings(self):
        """get_component_loadings()
        Return a square matrix of component loadings. Column j shows the amount
        of variance from each variable i in the original space that is accounted
        for by the jth principal component
        """
        loadings = np.zeros([self.dim, self.dim])
        if self.anal_type == "variance-covariance":
            std = np.sqrt(np.diag(self.Sigma))
        else:
            std = np.ones([self.dim])  # R's diagonals are 1, no need to compute
            
        for ax in range(self.dim):
            try:
                loadings[:, ax] = self.eigvec[:, ax] * np.sqrt(self.eigval[ax]) / std
            except RuntimeWarning as e:
                print(e)
                
        return loadings
        
        
if __name__ == '__main__':
    
    plt.ion()  # Interactive plotting

    
    (data, mu, Sigma) = genData(500, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    h_data = ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlabel('$f_0$')
    ax.set_ylabel('$f_1$')
    ax.set_zlabel('$f_2$')
    
    
    # plt.show()
    
    pca = PCA(data)

    # Show the PCA directions 
    k = 10  # scale up unit vectors by k 
    v = pca.get_pca_directions()
    for idx in range(v.shape[1]):
        ax.quiver(0, 0, 0, k * v[idx, 0], k * v[idx, 1], k * v[idx, 2], 
                  color='xkcd:orange')
    
    # project onto 2d 
    vc_proj = pca.transform(data, 2)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    h_vc_proj = ax.scatter(vc_proj[:, 0], vc_proj[:, 1], color='xkcd:orange')
    
    
    print("Component loadings")
    print(pca.get_component_loadings())
    
    print("Amount of variance captured m <= N components")
    print(np.cumsum(pca.eigval) / np.sum(pca.eigval))
    plt.legend([h_data, h_vc_proj], ['Original data', '$\Sigma $projection'])    
    

    
    # repeat w/ autocorreatlion analysis
    pcaR = PCA(data, corr_anal=True)
    
    w = pcaR.get_pca_directions()
    for idx in range(v.shape[1]):
        ax.quiver(0, 0, 0, k * w[idx, 0], k * w[idx, 1], k * w[idx, 2], color='xkcd:lilac')
    
    # project onto 2d 
    r_proj = pcaR.transform(data, 2)
    h_r_proj = ax.scatter(r_proj[:, 0], r_proj[:, 1], color='xkcd:lilac')

    # Add legend.  $...$ enables LaTeX-like equation formatting
    plt.legend([h_data, h_vc_proj, h_r_proj], 
               ['Original data', '$\Sigma $projection', '$R$ projection'])    
    print("Autocorrealtion component loadings")
    print(pcaR.get_component_loadings())
    
    print("Amount of normalized variance captured m <= N components")
    print(np.cumsum(pcaR.eigval) / np.sum(pcaR.eigval))


    
    x = 3  # breakable line so our windows don't go away
    
