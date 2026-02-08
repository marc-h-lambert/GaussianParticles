###################################################################################
# Code from the algorithm described in NIPS 2025 paper:                           #
# "Variational Inference with Mixtures of Isotropic Gaussians"                    #
# Marguerite Petit-Talamon, Marc Lambert, Anna Korba                              #
###################################################################################
# Fully vectorized version for efficient Gaussian Particles Optimization          #
###################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
from Core.LangevinTarget import GMM
from Core.GaussianParticles import GaussianParticles, VI_GaussianParticles
from decimal import getcontext

# Set global precision for Decimal
getcontext().prec = 6

# ==========================================
# Generate uniform GMM on a square domain
# ==========================================
def generate_uniform_GMM_square(xmin, xmax, ymin, ymax, N):
    """
    Generate a uniform grid of isotropic Gaussian particles over a 2D square.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Domain limits.
    N : int
        Number of Gaussians per axis (total K = N*N).

    Returns
    -------
    gp : GaussianParticles instance
    gmm : GMM instance
    """
    # Grid spacing
    dx = (xmax - xmin) / N
    dy = (ymax - ymin) / N

    # Grid of Gaussian means centered in each cell
    xs = np.linspace(xmin + dx / 2, xmax - dx / 2, N)
    ys = np.linspace(ymin + dy / 2, ymax - dy / 2, N)
    listMean = [np.array([x, y]) for y in ys for x in xs]

    # Covariances: isotropic with variance ~ dx^2, dy^2
    vars = np.ones(N * N) * dx ** 2

    gp = GaussianParticles(np.array(listMean), vars)
    return gp

# ==========================================
# Plot GMM and VI Gaussian Particles
# ==========================================
def plot_gmm(vgmm, target, Rgrid, stepPlot, num, Name="VI", Npoints=100):
    """
    Plot a sequence of GMMs over iterations for visual inspection.

    Parameters
    ----------
    vgmm : VI_GaussianParticles instance
        The variational Gaussian particles object.
    target : ImageTarget or similar
        The true target distribution for reference.
    Rgrid : float
        Range around target mean to plot.
    stepPlot : int
        Step interval for plotting intermediate steps.
    num : int
        Figure number.
    Name : str
        Plot title.
    Npoints : int
        Grid points for density approximation in the final plot.
    """
    center = target.mean()
    xmin, xmax = center[0] - Rgrid, center[0] + Rgrid
    ymin, ymax = center[1] - Rgrid, center[1] + Rgrid

    Ncols = 4
    step = 0
    fig, axs = plt.subplots(1, Ncols, figsize=(3 * Ncols, 4), num=num)
    fig.suptitle(Name)

    # Plot intermediate steps
    for i in range(Ncols - 1):
        ax = axs[i]
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_aspect("equal")

        gmm = vgmm.traj_gmm[step]
        target.plot(ax)
        gmm.plotCovs(ax)
        ax.set_title(f"Step N°{step}", fontsize=16)
        step += stepPlot

    # Plot final approximated density
    ax = axs[-1]
    gmm = vgmm.traj_gmm[-1]
    gmm.ComputeGrid2D(Rgrid, Npoints, center=center)
    gmm.plot(ax)
    ax.set_title("Final step\n(Approximated density)", fontsize=16)

    plt.tight_layout()


# ==========================================
# Main execution
# ==========================================
if __name__ == "__main__":
    ########## Choose your test ################
    TEST = ["GaussianParticles_UniformPrior"]
    num = 0

    # -----------------------------
    # Test vectorization & gradient correctness
    # -----------------------------
    if "GaussianParticles_Validation" in TEST:
        N = 10  # number of Gaussians
        d = 2
        means = np.zeros([N, d])
        vars = np.random.randn(N, 1) ** 2
        weights = np.ones([N, 1]) / N
        listCov = []

        # Arrange Gaussians on a circle
        R = 10
        theta = 0
        delta = 2 * math.pi / N
        for i in range(N):
            means[i] = [R * math.cos(theta), R * math.sin(theta)]
            theta += delta
            listCov.append(np.identity(2) * vars[i])

        gmm0 = GMM(weights, means, GMM.cov2sqrt(listCov))
        gp0 = GaussianParticles(gmm0.means.copy(), vars)

        # Validate gradient and pdf
        sample = np.zeros([2, 1])
        grad_ref = gmm0.gradientOld(sample)
        grad_gmm = gmm0.gradient(sample)
        grad_gp = gp0.gradient(sample)
        print("grad_ref =", grad_ref)
        print("grad_gmm =", grad_gmm)
        print("grad_gp =", grad_gp)

        pdf_gmm = gmm0.pdf(sample)
        pdf_gp = gp0.pdf(sample)
        print("pdf_gmm =", pdf_gmm)
        print("pdf_gp =", pdf_gp)

        # Validate batch gradient
        M = 100
        vars = vars.reshape(-1,)
        eps_samples = np.random.randn(N, M, d)
        samples_Batch_Mean = means[:, None, :] + np.sqrt(vars)[:, None, None] * eps_samples
        samples_Batch = samples_Batch_Mean.reshape(-1, d)

        grad_gmm_mean = gmm0.gradient(samples_Batch).mean(axis=0)
        grad_gp_mean = gp0.gradient(samples_Batch).mean(axis=0)
        print("grad_gmm_mean =", grad_gmm_mean)
        print("grad_gp_mean =", grad_gp_mean)

        pdf_gmm_mean = gmm0.pdf(samples_Batch).mean()
        pdf_gp_mean = gp0.pdf(samples_Batch).mean()
        print("pdf_gmm_mean =", pdf_gmm_mean)
        print("pdf_gp_mean =", pdf_gp_mean)

    # -----------------------------
    # Test approximation of a bimodal distribution
    # -----------------------------
    if "GaussianParticles_UniformPrior" in TEST:
        print('----------------------------------------------')
        print("TEST: Approximation of a bimodal distribution using Gaussian Particles")
        print('----------------------------------------------')

        # Setup target
        listMean = np.array([[5, 3.3], [5, -3.3]])
        listCov = np.array([np.diag([3, 1]), np.diag([1, 3])])
        listw = np.array([0.5, 0.5])
        target = GMM(listw, np.asarray(listMean), GMM.cov2sqrt(listCov))
        Npoints = 100
        Rgrid = 10
        target.ComputeGrid2D(Rgrid, Npoints=Npoints)
        center = target.mean()

        # Domain for initial GMM
        xmin, xmax = center[0] - 10, center[0] + 10
        ymin, ymax = center[1] - 10, center[1] + 10
        K = 10
        T = 10
        stepRK = 0.01
        stepPlot = 30

        # Initialize Gaussian Particles
        gp0 = generate_uniform_GMM_square(xmin, xmax, ymin, ymax, K)

        # Variational Inference
        vgp = VI_GaussianParticles(target, gp0, invbeta=1)
        vgp.propagate(stepRK, T)

        # Plot results
        plot_gmm(vgp, target, Rgrid, stepPlot, num, Name="VI GP")
        plt.show()
