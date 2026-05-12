import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
    VariationalStrategy,
)

from gpytorch_qr.mtgpqr import MultitaskALDLikelihood, MultitaskQuantileGP


def test_mtgpqr():
    def mean(x):
        return torch.cos(x * 2 * 3.14)

    def std(x):
        return x + 0.1

    x_range = torch.linspace(0, 1, 100).reshape(-1, 1)
    x = x_range.repeat(5, 1)
    y = (mean(x) + torch.randn(x.shape).mul(std(x))).squeeze()
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])

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
    gp = MyGP(inducing_points, len(q), num_latents=7)
    likelihood = MultitaskALDLikelihood(q)

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
        gp.mean_quantiles(x_pred)
