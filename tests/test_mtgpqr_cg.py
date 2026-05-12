import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from gpytorch_qr.mtgpqr_cg import (
    CenterGapLmcVariationalStrategy,
    MultitaskCenterGapALDLikelihood,
    MultitaskCenterGapQuantileGP,
)


def test_mtgpqr_cg():
    def mean(x):
        return torch.cos(x * 2 * 3.14)

    def std(x):
        return x + 0.1

    x_range = torch.linspace(0, 1, 100).reshape(-1, 1)
    x = x_range.repeat(5, 1)
    y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])

    class MyGP(MultitaskCenterGapQuantileGP):
        def __init__(
            self,
            inducing_points,
            num_quantiles,
            num_lower_quantiles,
            num_latents,
            num_lower_latents,
        ):
            N, D = inducing_points.size()
            variational_distribution = CholeskyVariationalDistribution(
                N,
                batch_shape=torch.Size([num_latents]),
            )
            variational_strategy = CenterGapLmcVariationalStrategy(
                VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                ),
                num_quantiles=num_quantiles,
                num_latents=num_latents,
                num_lower_quantiles=num_lower_quantiles,
                num_lower_latents=num_lower_latents,
            )

            center_mean = ConstantMean()
            gap_mean = ConstantMean(batch_shape=torch.Size([num_latents - 1]))
            covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=D, batch_shape=torch.Size([num_latents])),
                batch_shape=torch.Size([num_latents]),
            )
            super().__init__(
                variational_strategy,
                center_mean,
                gap_mean,
                covar_module,
                num_lower_quantiles,
            )

    inducing_points = torch.linspace(0, 1, 10).reshape(-1, 1)
    central_q_index = 2
    num_latents = 7
    gp = MyGP(inducing_points, len(q), central_q_index, num_latents, num_latents // 2)
    likelihood = MultitaskCenterGapALDLikelihood(q, central_q_index)

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
        gp.mean_quantiles_mc(x_pred, num_samples=1)
