###################################################################################
# LangevinTarget and GMM classes for variational inference                        #
# Supports vectorized Gaussian mixture gradients and grid evaluations             #                                    #
###################################################################################

import numpy as np
import math
import numpy.linalg as LA
from scipy.stats import multivariate_normal

class LangevinTarget:
    """
    Base class for target distributions used in Langevin dynamics or variational inference.
    Supports:
        - Grid evaluation of PDFs and gradients (2D only)
        - Sampling from precomputed PDF grids
        - Placeholder for PDF, log-PDF, and gradient methods
    """
    def __init__(self, Z=1):
        self.Z = Z
        self.empriricMean = None
        self.empiricalCov = None
        self.GridGradX = None
        self.GridGradY = None
        self.Gridpdf = None

    def mean(self):
        """Return mean of the target. Should be implemented in subclass."""
        return

    def pdf(self, x):
        """Return PDF value at x. To be implemented in subclass."""
        return

    def logpdf(self, x):
        """Return log-PDF value at x. To be implemented in subclass."""
        return

    def pdfUnnormalized(self, x):
        """Return unnormalized PDF (for grid computation)."""
        return self.pdf(x)

    def random(self):
        """
        Sample randomly from the grid-based PDF.
        Only works if ComputeGrid2D has been called.
        """
        if self.Gridpdf is None:
            raise ValueError("Random function unavailable: compute grid first.")
        nx, ny = self.Gridpdf.shape
        weights = self.Gridpdf.reshape(nx * ny,)
        weights /= weights.sum()
        idx = np.random.choice(np.arange(nx * ny), 1, p=weights)
        i = int(idx / nx)
        j = idx % nx
        theta = np.zeros((2, 1))
        theta[0] = self.xv[i, j]
        theta[1] = self.yv[i, j]
        return theta

    def gradient(self, theta):
        """Gradient of log PDF: to be implemented in subclass."""
        return

    def ComputeGrid2D(self, radius, Npoints, center=None, log=False):
        """
        Compute a 2D grid of PDF or log-PDF values for visualization or sampling.

        Args:
            radius: float, half-width of the grid in each dimension
            Npoints: int, number of points per axis
            center: optional 2D center of the grid
            log: bool, compute log-PDF if True
        """
        if center is None:
            center = self.mean()
        if hasattr(self, 'd') and self.d != 2:
            raise ValueError("ComputeGrid2D only valid for 2D targets.")

        self.empriricMean = np.zeros((2, 1))
        self.empriricCov = np.zeros((2, 2))
        theta1 = np.linspace(center[0] - radius, center[0] + radius, Npoints)
        theta2 = np.linspace(center[1] - radius, center[1] + radius, Npoints)
        self.xv, self.yv = np.meshgrid(theta1, theta2)
        self.Gridpdf = np.zeros((Npoints, Npoints))

        for i in range(Npoints):
            for j in range(Npoints):
                theta = np.array([self.xv[i, j], self.yv[i, j]])  # <-- flat array
                if log:
                    self.Gridpdf[i, j] = self.logpdf(theta)  # returns scalar
                else:
                    self.Gridpdf[i, j] = self.pdfUnnormalized(theta)  # returns scalar

    def gradientField(self, radius, Npoints, center=None):
        """
        Compute 2D gradient field of the log-posterior over a grid.

        Args:
            radius: float, grid radius
            Npoints: int, number of points per axis
            center: optional 2D center
        """
        if center is None:
            center = self.mean()
        if hasattr(self, 'd') and self.d != 2:
            raise ValueError("gradientField only valid for 2D targets.")

        theta1 = np.linspace(center[0] - radius, center[0] + radius, Npoints)
        theta2 = np.linspace(center[1] - radius, center[1] + radius, Npoints)
        self.xvg, self.yvg = np.meshgrid(theta1, theta2)
        self.GridGradX = np.zeros_like(self.xvg)
        self.GridGradY = np.zeros_like(self.yvg)

        for i in range(Npoints):
            for j in range(Npoints):
                theta = np.zeros(5)
                theta[0] = self.xvg[i, j]
                theta[1] = self.yvg[i, j]
                theta[2:4] = 0  # velocity components
                theta[4] = 1.0  # intensity placeholder
                grad = self.gradient(theta).reshape([2, 1])
                self.GridGradX[i, j] = grad[0]
                self.GridGradY[i, j] = grad[1]

    def plot_grad(self, ax):
        """Plot 2D gradient field using quiver arrows."""
        if self.GridGradX is not None:
            scale = 0.1
            ax.quiver(
                self.xvg, self.yvg,
                scale * self.GridGradX,
                scale * self.GridGradY,
                color='red', pivot='mid', scale=5
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Gradient field of log-posterior")

    def plot(self, ax, n_contours=20):
        """Plot 2D PDF contours."""
        if self.Gridpdf is not None:
            ax.contourf(
                self.xv, self.yv, self.Gridpdf,
                levels=n_contours,
                cmap="viridis", zorder=1
            )


class GMM(LangevinTarget):
    """
    Gaussian Mixture Model target for Langevin dynamics.
    Supports isotropic or full-covariance Gaussians.
    """
    def __init__(self, listw, listMean, listSqrt):
        super().__init__()
        listMean = np.asarray(listMean)
        self.K = listMean.shape[0]
        self.d = listMean[0].shape[0]
        self.weights = np.array(listw, dtype=float)
        self.means = np.array(listMean, dtype=float)
        self.rootcovs = np.array(listSqrt, dtype=float)

    def mean(self):
        return self.means.mean(axis=0)

    @staticmethod
    def cov2sqrt(listCov):
        """Convert list of covariance matrices to their Cholesky factors."""
        return np.array([LA.cholesky(P) for P in listCov])

    def pdf(self, X):
        """Compute mixture PDF at points X (N,d)."""
        X = np.atleast_2d(X)
        N, d = X.shape
        y = np.zeros(N)
        for k in range(self.K):
            w = self.weights[k]
            mu = self.means[k]
            R = self.rootcovs[k]
            P = R @ R.T
            y += w * multivariate_normal.pdf(X, mean=mu, cov=P)
        return y

    def pdfUnnormalized(self, x):
        return self.pdf(x)

    def logpdf(self, x):
        val = self.pdf(x)
        return math.log(val) if val > 0 else -np.inf

    def gradient(self, X):
        """Compute gradient of log-PDF at X (vectorized for N samples)."""
        X = np.atleast_2d(X)
        N, d = X.shape
        grad = np.zeros((N, d))
        pdf_vals = self.pdf(X)

        for k in range(self.K):
            w = self.weights[k]
            mu = self.means[k]
            R = self.rootcovs[k]
            P = R @ R.T
            P_inv = LA.inv(P)
            e = X - mu
            n = np.atleast_1d(multivariate_normal.pdf(X, mean=mu, cov=P))
            grad -= w * (e @ P_inv.T) * n[:, None]

        grad /= pdf_vals[:, None]
        if grad.shape[0] == 1:
            grad = grad.reshape(d, 1)
        return grad
