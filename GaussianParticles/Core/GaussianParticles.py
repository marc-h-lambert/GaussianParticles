
###################################################################################
# Code from the algorithm described in NIPS 2025 paper:                           #
# "Variational Inference with Mixtures of Isotropic Gaussians"                    #
# Marguerite Petit-Talamon, Marc Lambert, Anna Korba                              #
###################################################################################
# Fully vectorized version for efficient Gaussian Particles Optimization          #
# code supported by ML                                                            #
###################################################################################
"""
Tensor shapes for the computation of Variational Gradients:

For High-dim d (machine learning):
MC (Monte Carlo) Method (high variance, no bias):
----------
samples:        (N, M, d)       # N components, M MC samples, dimension d
grad_pi:        (N, M, d)       # ∇ log π evaluated at each sample
grad_q:         (N, M, d)       # ∇ log q for each sample
grad_diff:      (N, M, d)       # grad_q - grad_pi
E_grad_means:   (N, d)          # mean over MC samples
E_grad_vars:    (N,)            # scalar product over MC samples

For Low dim d (2D image processing, 3D fluids dynamics, 6D aerospace):
RGH (Randomized-Gauss-Hermite) Method (low variance, low bias):
Quadrature rule : GH=2 --> GH ** d samples
(Still in progress...)
-----------
nodes_rgh:      (nPts, d)       # GH nodes with random shift
samples:        (N, nPts, d)    # N components expanded
grad_pi:        (N, nPts, d)    # ∇ log π
grad_q:         (N, nPts, d)    # ∇ log q
grad_diff:      (N, nPts, d)
grad_diff_weighted: (N, nPts, d) # multiplied by GH weights
E_grad_means:   (N, d)          # sum over nPts
E_grad_vars:    (N,)            # weighted scalar products
"""

import numpy as np
from Core.LangevinTarget import LangevinTarget
import matplotlib.patches as patches
from decimal import Decimal, getcontext

# ============================================
# GaussianParticles: stores isotropic Gaussian mixture
# ============================================
class GaussianParticles(LangevinTarget):
    def __init__(self, listMean, listVars):
        """
        Initialize GaussianParticles with means and isotropic variances.

        Args:
            listMean: (N,d) array of Gaussian means
            listVars: (N,) array of isotropic variances
        """
        super().__init__()
        self.N = listMean.shape[0]         # Number of Gaussian components
        self.d = listMean.shape[1]         # Dimensionality of each Gaussian
        self.means = listMean              # Shape: (N, d)
        self.vars = listVars.reshape(-1,)  # Shape: (N,)

    def pdf(self, X):
        """
        Compute PDF of the isotropic Gaussian mixture at given points.

        Args:
            X: (M, d) array of points

        Returns:
            y: (M,) array of mixture PDF values
        """
        # Ensure X is 2D
        if np.squeeze(X).ndim == 1:
            X = X.reshape(-1, self.d)

        M, d = X.shape

        # Compute squared distances to each mean: (M,N)
        diff = X[:, None, :] - self.means[None, :, :]  # (M, N, d)
        dist2 = np.sum(diff ** 2, axis=-1)            # (M, N)

        # Compute Gaussian PDFs for each component: (M,N)
        coeff = (2 * np.pi * self.vars[None, :]) ** (-d / 2)
        pdfs = coeff * np.exp(-0.5 * dist2 / self.vars[None, :])

        # Mixture PDF (uniform weights 1/N)
        y = np.mean(pdfs, axis=1)
        return y

    def gradient(self, X):
        """
        Gradient of log PDF with respect to x for isotropic Gaussian mixture.

        Args:
            X: (M,d) array of points

        Returns:
            grad: (M,d) array of gradients ∇ log p(x)
        """
        if np.squeeze(X).ndim == 1:
            X = X.reshape(1, self.d)

        M, d = X.shape
        vars = self.vars.reshape(-1,)

        # Differences to each mean: (M,K,d)
        diff = X[:, None, :] - self.means[None, :, :]

        # Gaussian PDFs per component
        dist2 = np.sum(diff ** 2, axis=-1)
        coeff = (2 * np.pi * vars[None, :]) ** (-d / 2)
        pdfs = coeff * np.exp(-0.5 * dist2 / vars[None, :])

        # Mixture PDF
        pdf_vals = np.mean(pdfs, axis=1)

        # Gradient: weighted sum over components
        grad = -np.mean(pdfs[:, :, None] * (diff / vars[None, :, None]), axis=1)
        grad /= pdf_vals[:, None]  # gradient of log pdf

        if M == 1:
            grad = grad.reshape(self.d, 1)

        return grad

    def pdfUnnormalized(self, x):
        """Alias for pdf, unnormalized PDF not needed for isotropic GMM."""
        return self.pdf(x)

    def plotCovs(self, ax, nsig=3.0, color='r', label=None):
        """
        Plot isotropic Gaussian covariances as circles.

        nsig : number of standard deviations (e.g. 1σ, 2σ, 3σ)
        """
        for i in range(self.N):
            m = self.means[i, :]  # (2,)
            var = self.vars[i]  # (2,)
            radius = nsig * np.sqrt(var)

            circ = patches.Circle(
                (m[0], m[1]),
                radius=radius,
                fill=False,
                edgecolor=color,
                linewidth=2,
                label=label if (label is not None and i == 0) else None
            )
            ax.add_patch(circ)

# ============================================
# VI_GaussianParticles: variational inference with isotropic GMM
# ============================================
class VI_GaussianParticles():
    def __init__(self, target, GaussianParticles, invbeta=1, nbSamples=100, GaussHermite=False):
        """
        Variational Inference over GaussianParticles using MC or RGH.

        Args:
            target: object with .gradient(x) method
            GaussianParticles: initial GaussianParticles
            invbeta: optional scaling
            nbMC: number of Monte Carlo samples per Gaussian
            GaussHermite: use Randomized Gauss-Hermite if True
        """
        self.target = target
        self.means = GaussianParticles.means.copy()
        self.vars = GaussianParticles.vars.copy()
        self.N = self.means.shape[0]
        self.d = self.means.shape[1]
        self.invbeta = invbeta
        self.traj_gmm = [GaussianParticles]
        self.time = 0
        self.nbSamples = nbSamples
        self.GaussHermite = GaussHermite

    # --------------------------------------------
    # Time propagation of GMM parameters
    # --------------------------------------------
    def propagate(self, dt, T):
        """
        Evolve GMM particles up to time T using stepForward.

        Args:
            dt: timestep
            T: final time

        Returns:
            traj_gmm: list of GaussianParticles at each step
        """
        while self.time < T:
            print(self.time)
            self.time += Decimal(dt)

            # Propagate one step
            self.gmm = self.stepForward(dt)

            # Save trajectory
            self.traj_gmm.append(self.gmm)

        return self.traj_gmm

    # --------------------------------------------
    # Monte Carlo expected gradients
    # --------------------------------------------
    def VariationalGradientsMC(self):
        """
        Compute expected gradients w.r.t GMM parameters using Monte Carlo.

        Returns:
            E_grad_means: (N,d) mean gradient per component
            E_grad_vars:  (N,) variance gradient per component
        """
        N, d = self.means.shape
        M = self.nbSamples

        # 1. Sample from each isotropic Gaussian
        eps_samples = np.random.randn(N, M, d)
        samples = self.means[:, None, :] + np.sqrt(self.vars)[:, None, None] * eps_samples
        samples_flat = samples.reshape(-1, d)

        # 2. Compute target gradient ∇ log π(x)
        grad_pi_flat = self.target.gradient(samples_flat)
        grad_pi = grad_pi_flat.reshape(N, M, d)

        # 3. GMM gradient for isotropic, uniform-weight components
        grad_q = -(samples - self.means[:, None, :]) / self.vars[:, None, None]

        # 4. Difference
        grad_diff = grad_q - grad_pi

        # 5. Expected gradient per component
        E_grad_means = grad_diff.mean(axis=1)

        # 6. Expected scalar product for variance update
        scalar_products = np.sum((samples - self.means[:, None, :]) * grad_diff, axis=2)
        E_grad_vars = scalar_products.mean(axis=1)

        return E_grad_means, E_grad_vars

    # --------------------------------------------
    # Randomized Gauss-Hermite expected gradients
    # --------------------------------------------
    def VariationalGradientsRGH(self):
        """
        Compute expected gradients w.r.t GMM parameters using Randomized Gauss-Hermite.

        Args:
            nGH: number of GH nodes per dimension

        Returns:
            E_grad_means: (N,d)
            E_grad_vars:  (N,)
        """
        from numpy.polynomial.hermite import hermgauss
        import itertools

        N, d = self.means.shape
        # We generate nGH**d samples
        nGH = max(1, int(round(self.nbSamples ** (1/self.d))))

        # 1D Gauss-Hermite nodes and weights
        nodes_1d, weights_1d = hermgauss(nGH)
        nodes_1d = nodes_1d.astype(float)
        weights_1d = weights_1d.astype(float)

        # Tensor-product nodes and weights for d dimensions
        nodes_list = list(itertools.product(nodes_1d, repeat=d))
        weights_list = list(itertools.product(weights_1d, repeat=d))
        nodes = np.array(nodes_list)
        weights = np.prod(np.array(weights_list), axis=1)  # product of weights
        nPts = nodes.shape[0]

        # Randomized shift in [-0.5, 0.5] per dimension
        shift = np.random.uniform(-0.5, 0.5, size=(nPts, d))
        nodes_rgh = nodes + shift

        # Expand per Gaussian component: x = mean + sqrt(2*var)*node
        sqrt2var = np.sqrt(2.0 * self.vars)[:, None]
        samples = self.means[:, None, :] + nodes_rgh[None, :, :] * sqrt2var[:, None, :]

        # Flatten for target gradient
        samples_flat = samples.reshape(N * nPts, d)
        grad_pi_flat = self.target.gradient(samples_flat)
        grad_pi = grad_pi_flat.reshape(N, nPts, d)

        # GMM gradient
        grad_q = -(samples - self.means[:, None, :]) / self.vars[:, None, None]

        # Weighted difference
        grad_diff = grad_q - grad_pi
        grad_diff_weighted = grad_diff * weights[None, :, None]

        # Expected gradient per component
        E_grad_means = grad_diff_weighted.sum(axis=1)

        # Expected scalar products for variance update
        diff = samples - self.means[:, None, :]
        scalar_products_weighted = np.sum(diff * grad_diff * weights[None, :, None], axis=2)
        E_grad_vars = scalar_products_weighted.sum(axis=1)

        return E_grad_means, E_grad_vars

    # --------------------------------------------
    # Propagation step
    # --------------------------------------------
    def stepForward(self, dt):
        """
        One propagation step for isotropic GMM particles.

        Args:
            dt: timestep

        Returns:
            GaussianParticles instance with updated means and variances
        """
        # Choose gradient method
        if self.GaussHermite:
            E_grad_means, E_grad_vars = self.VariationalGradientsRGH()
        else:
            E_grad_means, E_grad_vars = self.VariationalGradientsMC()

        # Update rules
        self.means -= dt * E_grad_means
        self.vars *= (1 - dt * E_grad_vars / (self.d * self.vars)) ** 2

        return GaussianParticles(self.means.copy(), self.vars.copy())
