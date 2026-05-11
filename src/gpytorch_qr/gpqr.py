"""GPQR with each quantile as independent batch GP.

.. code-block:: python
   :caption: Example
   :linenos:

    import torch
    from torch.distributions import Normal

    torch.manual_seed(42)

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
    from gpytorch.variational import VariationalStrategy
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch_qr.gpqr import BatchQuantileGP, BatchALDLikelihood

    class MyGP(BatchQuantileGP):
        def __init__(self, inducing_points, num_quantiles):
            N, D = inducing_points.size()
            variational_distribution = CholeskyVariationalDistribution(
                N,
                batch_shape=torch.Size([num_quantiles]),
            )
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )
            mean = ConstantMean(batch_shape=torch.Size([num_quantiles]))
            covar = ScaleKernel(
                RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_quantiles])),
                batch_shape=torch.Size([num_quantiles]),
            )
            super().__init__(variational_strategy, mean, covar)

    inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
    gp = MyGP(inducing_points, len(q))
    likelihood = BatchALDLikelihood(q)

    from gpytorch.mlls import VariationalELBO

    gp.train()
    likelihood.train()
    mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    optimizer = torch.optim.Adam(
        list(gp.parameters()) + list(likelihood.parameters()),
        lr=0.001,
    )

    for _ in range(1000):
        output = gp(x)
        loss = -mll(output, y).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gp.eval()
    x_pred = torch.linspace(0, 2, 100).reshape(-1, 1)
    with torch.no_grad():
        q_posterior = gp.marginal_posterior(x_pred)

    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')
    plt.plot(x_pred, q_posterior.mean.T)
"""

import gpytorch
import torch

from .ald import BatchALD

__all__ = [
    "BatchQuantileGP",
    "BatchALDLikelihood",
]


class BatchQuantileGP(gpytorch.models.ApproximateGP):
    """Batch approximate GP for multiple quantiles.

    Parameters
    ----------
    variational_strategy : gpytorch.variational.VariationalStrategy
        The variational strategy.
    mean_module : gpytorch.means.Mean
        Mean module with batch shape equal to the number of quantiles.
    covar_module : gpytorch.kernels.Kernel
        Covariance module with batch shape equal to the number of quantiles.
    """

    def __init__(self, variational_strategy, mean_module, covar_module):
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def marginal_posterior(self, x):
        """Marginal posterior over quantiles.

        Parameters
        ----------
        x : torch.Tensor with shape (N, D)
            The input locations.

        Returns
        -------
        distribution : torch.distributions.Normal
            Marginal posterior over quantiles at input locations.
            ``loc`` has shape (Q, N) and ``scale`` has shape (Q, N),
            where *Q* is the number of quantiles and *N* is the number of data points.

        Notes
        -----
        To obtain the full joint posterior over quantiles, use ``self(x)``.
        """
        dist = self(x)
        return torch.distributions.Normal(dist.mean, dist.variance.sqrt())


class BatchALDLikelihood(gpytorch.likelihoods.Likelihood):
    """ALD likelihood for batch quantile regression.

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
        function_samples : torch.Tensor with shape (S, Q, N)
            The function samples drawn from the posterior of quantile functions.
            *S* is the number of samples, *Q* is the number of quantiles,
            and *N* is the number of data points.
        """
        return BatchALD(
            m=function_samples,  # (S, Q, N)
            lamda=self.scales,  # (Q,)
            kappa=self.q,  # (Q,)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        # lp: (Q, N)
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=0)  # (N,)
