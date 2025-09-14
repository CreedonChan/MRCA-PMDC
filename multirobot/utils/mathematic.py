import numpy as np
import torch


def smallest_angle_diff_np(target, source):
    diff = np.arctan2(np.sin(target - source), np.cos(target - source))
    return diff


def angle_between_2d_vectors(v1, v2):
    v1 = v1.float()
    v2 = v2.float()

    if v1.dim() == 1:
        v1 = v1.unsqueeze(0)
        v2 = v2.unsqueeze(0)

    dot_product = (v1 * v2).sum(dim=-1)

    norm_v1 = torch.norm(v1, dim=-1)
    norm_v2 = torch.norm(v2, dim=-1)

    eps = 1e-8
    norm_product = norm_v1 * norm_v2 + eps

    cos_theta = dot_product / norm_product
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angle = torch.acos(cos_theta)

    return angle.squeeze()


def angle_between_3d_vectors(v1, v2):
    v1 = v1.float()
    v2 = v2.float()

    if v1.dim() == 1:
        v1 = v1.unsqueeze(0)
        v2 = v2.unsqueeze(0)

    dot_product = (v1 * v2).sum(dim=-1)

    norm_v1 = torch.norm(v1, dim=-1)
    norm_v2 = torch.norm(v2, dim=-1)

    eps = 1e-8
    norm_product = norm_v1 * norm_v2 + eps

    cos_theta = dot_product / norm_product
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angle = torch.acos(cos_theta)

    return angle.squeeze()


def xy_to_magnitude_angle(tensor):
    assert tensor.size(-1) == 2
    x, y = tensor[..., 0], tensor[..., 1]
    magnitude = torch.sqrt(x ** 2 + y ** 2)
    angle = torch.atan2(y, x)
    return torch.stack([magnitude, angle], dim=-1)


def magnitude_angle_to_xy(tensor):
    assert tensor.size(-1) == 2
    magnitude, angle = tensor[..., 0], tensor[..., 1]
    x = magnitude * torch.cos(angle)
    y = magnitude * torch.sin(angle)
    return torch.stack([x, y], dim=-1)


def update_mean_cov_single(mu, C, n, new_point):
    mu_new = mu + (new_point - mu) / (n + 1)
    delta = new_point - mu
    C_new = (n / (n + 1)) * C + (n / (n + 1) ** 2) * np.outer(delta, delta)
    return mu_new, C_new, n + 1


def update_mean_cov_single_unbiased(mu, C, n, new_point):
    mu_new = mu + (new_point - mu) / (n + 1)
    delta = new_point - mu
    C_new = ( (n - 1) / n ) * C + (1 / n) * np.outer(delta, delta)
    return mu_new, C_new, n + 1


def compute_vector_expectation_and_covariance(samples):
    X = np.vstack(samples)
    n, d = X.shape
    expectation = np.mean(X, axis=0)
    X_centered = X - expectation
    covariance = (X_centered.T @ X_centered) / n
    return expectation, covariance


def compute_vector_expectation(samples):
    X = np.vstack(samples)
    expectation = np.mean(X, axis=0)
    return expectation


def gershgorin_lower_bound(A):
    assert A.dim() == 3, "Input should be a tensor of shape (batch_size, n, n)"
    R = torch.sum(torch.abs(A), dim=2) - torch.abs(torch.diagonal(A, dim1=1, dim2=2))
    diag = torch.diagonal(A, dim1=1, dim2=2)
    aii_minus_Ri = diag - R
    lower_bounds = torch.max(aii_minus_Ri, dim=1).values
    return lower_bounds
