"""
1D regression dataset with heteroskedastic noise:

.. plot::
   :context:

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
    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')

Define GP:

.. plot::
   :context: close-figs

    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import VariationalStrategy
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch_qr.gpqr import QuantileGP, ALDLikelihood

    class MyQuantileGP(QuantileGP):
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
    gp = MyQuantileGP(inducing_points, len(q))
    likelihood = ALDLikelihood(q)

Train the model:

.. plot::
   :context: close-figs

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

Evaluate:

.. plot::
   :context: close-figs

    gp.eval()
    with torch.no_grad():
        quantiles = gp(x_pred).mean.detach()

    plt.scatter(x, y, c='gray', marker='.', alpha=0.1)
    plt.plot(x_range, true_quantiles, '--', c='k')
    plt.plot(x_pred, quantiles)
"""

import gpytorch
import torch

__all__ = [
    "QuantileGP",
    "ALD",
    "ALDLikelihood",
]


class QuantileGP(gpytorch.models.ApproximateGP):
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


class ALD(torch.distributions.Distribution):
    """Batched asymmetric Laplace distribution for quantile regression.

    Parameters
    ----------
    m : torch.Tensor with shape (S, Q, N)
        The location parameters of the distribution.
    lamda : torch.Tensor with shape (Q,)
        The scale parameters of the distribution for each quantile.
    kappa : torch.Tensor with shape (Q,)
        The quantile levels of the distribution.

    Notes
    -----
    In the context of quantile regression, the location parameter *m* corresponds to
    sample points drawn from posterior distributions of quantile functions.
    For *Q* quantiles, *S* samples are drawn for *N* data points.

    The value passed to :meth:`log_prob` is the observed *y* values.
    """

    arg_constraints = {
        "m": torch.distributions.constraints.real,
        "lamda": torch.distributions.constraints.positive,
        "kappa": torch.distributions.constraints.unit_interval,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, m, lamda, kappa):
        self.m = m
        self.lamda = lamda.unsqueeze(-1)  # (Q, 1)
        self.kappa = kappa.unsqueeze(-1)  # (Q, 1)
        batch_shape = torch.broadcast_shapes(
            m.shape, self.lamda.shape, self.kappa.shape
        )
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([]))

    def log_prob(self, value):
        """Log probability of the asymmetric Laplace distribution at the given value.

        Parameters
        ----------
        value : torch.Tensor with shape (N,)
            The values at which to evaluate the log probability.

        Returns
        -------
        logp : torch.Tensor with shape (S, Q, N)
            The log probability at the given values for each quantile and sample.
        """
        # value: (N,)
        residual = value - self.m
        check = residual * (self.kappa - (residual < 0).to(residual))
        logp = (
            torch.log(self.kappa)
            + torch.log(1 - self.kappa)
            - torch.log(self.lamda)
            - check / self.lamda
        )  # (S, Q, N)
        return logp


class ALDLikelihood(gpytorch.likelihoods.Likelihood):
    """ALD likelihood for multiple quantile levels.

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
        # function_samples: (S, Q, N)
        # S: Number of MC samples.
        # Q: Number of quantiles.
        # N: Number of data points, i.e., length of x passed to gp(x).
        return ALD(
            m=function_samples,
            lamda=self.scales,  # (Q,)
            kappa=self.q,  # (Q,)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        # lp: (Q, N)
        lp = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return lp.sum(dim=0)  # (N,)
