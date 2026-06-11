import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
    VariationalStrategy,
)

from gpytorch_qr.likelihoods import MultitaskQuantileGPLikelihood
from gpytorch_qr.models import DirectQuantileGP


def test_mtgpqr():
    def mean(x):
        return torch.cos(x * 2 * 3.14)

    def std(x):
        return x + 0.1

    x_range = torch.linspace(0, 1, 10).reshape(-1, 1)
    x = x_range.repeat(2, 1)
    y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])

    class MyGP(DirectQuantileGP):
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
    gp = MyGP(inducing_points, len(q), num_latents=7)
    likelihood = MultitaskQuantileGPLikelihood(q)

    gp.train()
    likelihood.train()
    mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    optimizer = torch.optim.Adam(
        list(gp.parameters()) + list(likelihood.parameters()),
        lr=0.01,
    )

    for _ in range(1):
        output = gp(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gp.eval()
    x_pred = torch.linspace(0, 2, 5).reshape(-1, 1)
    with torch.no_grad():
        gp.joint_quantile_posterior(x_pred)
        gp.marginal_quantile_posterior(x_pred)
        gp.mean_quantiles(x_pred)
        gp.mean_quantiles_mc(x_pred, num_samples=1)
        gp.mean_quantiles_delta(x_pred)
        gp.quantile_quantiles(x_pred, torch.tensor([0.025, 0.975]))
        gp.quantile_quantiles_mc(x_pred, torch.tensor([0.025, 0.975]), num_samples=1)
        likelihood.predictive_posterior(gp(x_pred))


def test_mtgpqr_multivariate():
    def mean(x):
        return torch.cos(x[:, 0] * 2 * 3.14) * torch.cos(x[:, 1] * 2 * 3.14)

    def std(x):
        return x[:, 0] + x[:, 1] + 0.1

    x2_values = torch.tensor([0.1, 0.5])
    n_per_x2 = 2
    x = torch.stack(
        [
            torch.rand(n_per_x2 * len(x2_values)),
            x2_values.repeat_interleave(n_per_x2),
        ],
        dim=1,
    )
    y = (mean(x) + torch.randn(x.shape[0]).mul(std(x))).squeeze()
    q = torch.tensor([0.1, 0.5, 0.9])

    class MyGP(DirectQuantileGP):
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

    g1, g2 = torch.meshgrid(
        torch.linspace(0, 1, 2),
        torch.tensor([0.1, 0.5]),
        indexing="ij",
    )
    inducing_points = torch.stack([g1.flatten(), g2.flatten()], dim=1)
    gp = MyGP(inducing_points, len(q), num_latents=3)
    likelihood = MultitaskQuantileGPLikelihood(q)

    gp.train()
    likelihood.train()
    mll = VariationalELBO(likelihood, gp, num_data=y.numel())
    optimizer = torch.optim.Adam(
        list(gp.parameters()) + list(likelihood.parameters()),
        lr=0.01,
    )

    for _ in range(1):
        output = gp(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gp.eval()
    x_pred = torch.tensor([[0.0, 0.1], [1.0, 0.5]])
    with torch.no_grad():
        gp.joint_quantile_posterior(x_pred)
        gp.marginal_quantile_posterior(x_pred)
        gp.mean_quantiles(x_pred)
        gp.mean_quantiles_mc(x_pred, num_samples=1)
        gp.mean_quantiles_delta(x_pred)
        gp.quantile_quantiles(x_pred, torch.tensor([0.025, 0.975]))
        gp.quantile_quantiles_mc(x_pred, torch.tensor([0.025, 0.975]), num_samples=1)
        likelihood.predictive_posterior(gp(x_pred))
