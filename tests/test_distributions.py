import torch

from gpytorch_qr.distributions import AsymmetricLaplace


def test_asymmetric_laplace_log_prob():
    distribution = AsymmetricLaplace(
        loc=torch.tensor(1.0),
        scale=torch.tensor(2.0),
        asymmetry=torch.tensor(3.0),
    )
    value = torch.tensor([-1.0, 1.0, 2.0])

    expected = torch.tensor(
        [
            torch.log(torch.tensor(6.0 / 10.0)) - 4.0 / 3.0,
            torch.log(torch.tensor(6.0 / 10.0)),
            torch.log(torch.tensor(6.0 / 10.0)) - 6.0,
        ]
    )
    assert torch.allclose(distribution.log_prob(value), expected)


def test_asymmetric_laplace_cdf_and_icdf_are_inverses():
    distribution = AsymmetricLaplace(
        loc=torch.tensor(1.0),
        scale=torch.tensor(2.0),
        asymmetry=torch.tensor(3.0),
    )
    probabilities = torch.tensor([0.01, 0.5, 0.9, 0.99])

    assert torch.allclose(
        distribution.cdf(distribution.icdf(probabilities)), probabilities
    )
    assert torch.allclose(distribution.cdf(torch.tensor(1.0)), torch.tensor(0.9))


def test_asymmetric_laplace_rsample_shape_and_gradients():
    loc = torch.tensor([0.0, 1.0], requires_grad=True)
    scale = torch.tensor([1.0, 2.0], requires_grad=True)
    asymmetry = torch.tensor([0.5, 2.0], requires_grad=True)
    distribution = AsymmetricLaplace(loc, scale, asymmetry)

    samples = distribution.rsample(torch.Size([4, 3]))
    samples.sum().backward()

    assert samples.shape == torch.Size([4, 3, 2])
    assert loc.grad is not None
    assert scale.grad is not None
    assert asymmetry.grad is not None
