"""
PyTorch port of pyStarlet_2D1D_jax.py

Key design decision: For symmetric padding we use numpy.pad (mode="symmetric"),
which is numerically identical to JAX's jnp.pad(mode="symmetric") on CPU.
The result is then converted back to a torch tensor. This keeps the math simple
and avoids custom index-arithmetic bugs.

All other operations (b-spline smoothing, subtraction, stacking) are native torch ops.
"""
import torch
import numpy as np

############################################################
################# STARLET TRANSFORM
############################################################


class StarletError(Exception):
    pass


class WrongDimensionError(StarletError):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "The data has a wrong number of dimension."


##############################################################################
# Padding helpers: delegate to numpy for exact semantic match with JAX.

def _sym_pad(t: torch.Tensor, pad_config: tuple) -> torch.Tensor:
    """
    Pure-torch symmetric padding — identical to jnp.pad(..., mode='symmetric').
    Stays on the original device (MPS / CUDA / CPU) with no numpy round-trip.

    `pad_config` : sequence of (pad_before, pad_after) per axis, same as numpy.pad.
    """
    for dim, (pad_before, pad_after) in enumerate(pad_config):
        if pad_before == 0 and pad_after == 0:
            continue
        N = t.size(dim)
        # Build the full output index range for this dimension
        idx = torch.arange(-pad_before, N + pad_after, device=t.device)
        # Symmetric fold: idx mod 2N, then mirror upper half
        idx = idx % (2 * N)
        idx = torch.where(idx >= N, 2 * N - 1 - idx, idx)
        t = torch.index_select(t, dim, idx)
    return t


##############################################################################
# B-spline smoothing kernels (pure torch; no padding needed here)

def smooth_bspline(padded_img: torch.Tensor, step_trou: int, pad: int) -> torch.Tensor:
    """2D b-spline, input is already padded."""
    h0, h1, h2 = 3./8., 1./4., 1./16.
    step = int(pow(2., step_trou) + 0.5)

    buff = (h0 * padded_img[pad:-pad, pad:-pad]
          + h1 * padded_img[pad:-pad, pad-step:-pad-step]
          + h1 * padded_img[pad:-pad, pad+step:-pad+step]
          + h2 * padded_img[pad:-pad, pad-2*step:-pad-2*step]
          + h2 * padded_img[pad:-pad, pad+2*step:-pad+2*step])

    # Re-pad buff before the second separable pass
    buff = _sym_pad(buff, ((pad, pad), (pad, pad)))

    img_out = (h0 * buff[pad:-pad, pad:-pad]
             + h1 * buff[pad-step:-pad-step, pad:-pad]
             + h1 * buff[pad+step:-pad+step, pad:-pad]
             + h2 * buff[pad-2*step:-pad-2*step, pad:-pad]
             + h2 * buff[pad+2*step:-pad+2*step, pad:-pad])
    return img_out


def smooth_bspline2D_forcube(padded_cube: torch.Tensor, step_trou: int, pad: int) -> torch.Tensor:
    """2D b-spline applied identically to each Z-slice of a (Z, X, Y) padded cube."""
    h0, h1, h2 = 3./8., 1./4., 1./16.
    step = int(pow(2., step_trou) + 0.5)

    buff = (h0 * padded_cube[:, pad:-pad, pad:-pad]
          + h1 * padded_cube[:, pad:-pad, pad-step:-pad-step]
          + h1 * padded_cube[:, pad:-pad, pad+step:-pad+step]
          + h2 * padded_cube[:, pad:-pad, pad-2*step:-pad-2*step]
          + h2 * padded_cube[:, pad:-pad, pad+2*step:-pad+2*step])

    buff = _sym_pad(buff, ((0, 0), (pad, pad), (pad, pad)))

    img_out = (h0 * buff[:, pad:-pad, pad:-pad]
             + h1 * buff[:, pad-step:-pad-step, pad:-pad]
             + h1 * buff[:, pad+step:-pad+step, pad:-pad]
             + h2 * buff[:, pad-2*step:-pad-2*step, pad:-pad]
             + h2 * buff[:, pad+2*step:-pad+2*step, pad:-pad])
    return img_out


def smooth_bspline1D(padded_vec: torch.Tensor, step_trou: int, pad: int) -> torch.Tensor:
    """1D b-spline applied to a 1D padded vector."""
    h0, h1, h2 = 3./8., 1./4., 1./16.
    step = int(pow(2., step_trou) + 0.5)
    return (h0 * padded_vec[pad:-pad]
          + h1 * padded_vec[pad-step:-pad-step]
          + h1 * padded_vec[pad+step:-pad+step]
          + h2 * padded_vec[pad-2*step:-pad-2*step]
          + h2 * padded_vec[pad+2*step:-pad+2*step])


def smooth_bspline1D_forcube(padded_cube: torch.Tensor, step_trou: int, pad: int) -> torch.Tensor:
    """1D b-spline along axis-0 of a (Z, X, Y) padded cube."""
    h0, h1, h2 = 3./8., 1./4., 1./16.
    step = int(pow(2., step_trou) + 0.5)
    return (h0 * padded_cube[pad:-pad, :, :]
          + h1 * padded_cube[pad-step:-pad-step, :, :]
          + h1 * padded_cube[pad+step:-pad+step, :, :]
          + h2 * padded_cube[pad-2*step:-pad-2*step, :, :]
          + h2 * padded_cube[pad+2*step:-pad+2*step, :, :])


def smooth_bspline1D_forhypercube(padded_hcube: torch.Tensor, step_trou: int, pad: int) -> torch.Tensor:
    """1D b-spline along axis-1 (Z) of a (N, Z, X, Y) padded hypercube."""
    h0, h1, h2 = 3./8., 1./4., 1./16.
    step = int(pow(2., step_trou) + 0.5)
    return (h0 * padded_hcube[:, pad:-pad, :, :]
          + h1 * padded_hcube[:, pad-step:-pad-step, :, :]
          + h1 * padded_hcube[:, pad+step:-pad+step, :, :]
          + h2 * padded_hcube[:, pad-2*step:-pad-2*step, :, :]
          + h2 * padded_hcube[:, pad+2*step:-pad+2*step, :, :])


def mad(z: torch.Tensor) -> torch.Tensor:
    return torch.median(torch.abs(z - torch.median(z))) / 0.6735


##############################################################################
# Forward transforms

def Starlet_Forward2D(input_image: torch.Tensor, J: int, M: int, N: int):
    image = input_image.clone()
    pad = 2 * int(pow(2., J) + 0.5) + 1
    planes = []
    planes.append(image)

    for scale_index in range(J):
        prev = planes[scale_index]
        padded = _sym_pad(prev, ((pad, pad), (pad, pad)))
        nxt = smooth_bspline(padded, scale_index, pad)
        planes[scale_index] = prev - nxt
        planes.append(nxt)

    coarse = planes[-1]
    return coarse, torch.stack(planes[:-1])


def Starlet_Forward2D_for_cube(input_image: torch.Tensor, J: int):
    """2D Starlet on each slice of a (Z, X, Y) cube."""
    image = input_image.clone()
    pad = 2 * int(pow(2., J) + 0.5) + 1
    planes = [image]

    for scale_index in range(J):
        prev = planes[scale_index]
        padded = _sym_pad(prev, ((0, 0), (pad, pad), (pad, pad)))
        nxt = smooth_bspline2D_forcube(padded, scale_index, pad)
        planes[scale_index] = prev - nxt
        planes.append(nxt)

    coarse = planes[-1]
    return coarse, torch.stack(planes[:-1])


def Starlet_Forward1D(input_image: torch.Tensor, J: int, L: int):
    image = input_image.clone()
    pad = 2 * int(pow(2., J) + 0.5) + 1
    planes = [image]

    for scale_index in range(J):
        prev = planes[scale_index]
        padded = _sym_pad(prev, ((pad, pad),))
        nxt = smooth_bspline1D(padded, scale_index, pad)
        planes[scale_index] = prev - nxt
        planes.append(nxt)

    coarse = planes[-1]
    return coarse, torch.stack(planes[:-1])


def Starlet_Forward1D_forcube(input_image: torch.Tensor, J: int):
    """1D Starlet along axis-0 (Z) of a (Z, X, Y) cube."""
    image = input_image.clone()
    pad = 2 * int(pow(2., J) + 0.5) + 1
    planes = [image]

    for scale_index in range(J):
        prev = planes[scale_index]
        padded = _sym_pad(prev, ((pad, pad), (0, 0), (0, 0)))
        nxt = smooth_bspline1D_forcube(padded, scale_index, pad)
        planes[scale_index] = prev - nxt
        planes.append(nxt)

    coarse = planes[-1]
    return coarse, torch.stack(planes[:-1])


def Starlet_Forward1D_forhypercube(input_image: torch.Tensor, J: int):
    """1D Starlet along axis-1 (Z) of a (N, Z, X, Y) hypercube."""
    image = input_image.clone()
    pad = 2 * int(pow(2., J) + 0.5) + 1
    planes = [image]

    for scale_index in range(J):
        prev = planes[scale_index]
        padded = _sym_pad(prev, ((0, 0), (pad, pad), (0, 0), (0, 0)))
        nxt = smooth_bspline1D_forhypercube(padded, scale_index, pad)
        planes[scale_index] = prev - nxt
        planes.append(nxt)

    coarse = planes[-1]
    return coarse, torch.stack(planes[:-1])


def Starlet_Forward2D_1D(input_image: torch.Tensor, J_1D: int = 3, J_2D: int = 2):
    """Combined 2D+1D Starlet transform matching pyStarlet_2D1D_jax.Starlet_Forward2D_1D."""
    image = input_image.clone()
    c2D, w2D = Starlet_Forward2D_for_cube(image, J_2D)
    cc_2D1D, cw_2D1D = Starlet_Forward1D_forcube(c2D, J_1D)
    wc_2D1D, ww_2D1D = Starlet_Forward1D_forhypercube(w2D, J_1D)
    return cc_2D1D, cw_2D1D, wc_2D1D, ww_2D1D
