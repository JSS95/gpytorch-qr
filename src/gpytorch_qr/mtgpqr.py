"""Multitask GPQR.

Latent GPs directly construct quantiles.

.. plot::
   :context: reset
   :include-source: False

    import torch
    from torch.distributions import Normal

    def mean(x):
        return torch.cos(x * 2 * 3.14)

    def std(x):
        return x + 0.1

    x_range = torch.linspace(0, 1, 100).reshape(-1, 1)
    x = x_range.repeat(5, 1)
    y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    true_quantiles = mean(x_range) + std(x_range) * Normal(0, 1).icdf(q)

    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import VariationalStrategy, LMCVariationalStrategy
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch_qr.mtgpqr import MultitaskQuantileGP, MultitaskALDLikelihood

    class MyGP(MultitaskQuantileGP):
        def __init__(self, inducing_points, num_quantiles, num_latents):
            N, D = inducing_points.size()
            variational_distribution = CholeskyVariationalDistribution(
                N,
                batch_shape=torch.Size([num_latents]),
            )
            variational_strategy = LMCVariationalStrategy(
                VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                ),
                num_tasks=num_quantiles,
                num_latents=num_latents,
            )

            mean_module = ConstantMean(batch_shape=torch.Size([num_latents]))
            covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents]),
            )
            super().__init__(variational_strategy, mean_module, covar_module)

    inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
    gp = MyGP(inducing_points, len(q), num_latents=len(q))
    likelihood = MultitaskALDLikelihood(q)

    from gpytorch.mlls import VariationalELBO

    gp.train()
    likelihood.train()
    mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    optimizer = torch.optim.Adam(
        list(gp.parameters()) + list(likelihood.parameters()),
        lr=0.01,
    )

    for _ in range(100):
        output = gp(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gp.eval()
    x_pred = torch.linspace(0, 2, 100).reshape(-1, 1)
    with torch.no_grad():
        quantiles = gp.mean_quantiles(x_pred).detach()

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')
    plt.plot(x_pred, quantiles)
"""

import gpytorch
import torch

from .ald import MultitaskALD

__all__ = [
    "MultitaskQuantileGP",
    "MultitaskALDLikelihood",
]


class MultitaskQuantileGP(gpytorch.models.ApproximateGP):
    """Multitask approximate GP for multiple quantiles.

    Parameters
    ----------
    variational_strategy : gpytorch.variational.VariationalStrategy
        The variational strategy.
    mean_module : gpytorch.means.Mean
        Mean module with batch shape equal to the number of latent GPs.
    covar_module : gpytorch.kernels.Kernel
        Covariance module with batch shape equal to the number of latent GPs.
    """

    def __init__(self, variational_strategy, mean_module, covar_module):
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def mean_quantiles(self, x):
        """Predict quantiles by posterior mean.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.

        Returns
        -------
        quantiles : torch.Tensor with shape (N, Q)
            The predicted quantiles at the input locations.
        """
        return self(x).mean


class MultitaskALDLikelihood(gpytorch.likelihoods.Likelihood):
    """ALD likelihood for multitask quantile regression.

    Parameters
    ----------
    q : torch.Tensor with shape (Q,)
        The quantile levels.
    """

    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", q.float())
        self.register_parameter(
            "raw_scales",
            torch.nn.Parameter(torch.zeros(len(q))),
        )
        self.register_constraint("raw_scales", gpytorch.constraints.Positive())

    @property
    def scales(self):
        return self.raw_scales_constraint.transform(self.raw_scales)

    def forward(self, function_samples):
        """Return the ALD distribution for the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor with shape (S, N, Q)
            The function samples drawn from the posterior distributions of quantile
            functions. *S* is the number of samples, *N* is the number of data points,
            and *Q* is the number of quantiles.
        """
        return MultitaskALD(
            m=function_samples,  # (S, N, Q)
            lamda=self.scales,  # (Q,)
            kappa=self.q,  # (Q,)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        lp = super().expected_log_prob(
            observations, function_dist, *args, **kwargs
        )  # (N, Q)
        return lp.sum(dim=1)  # (N,)
