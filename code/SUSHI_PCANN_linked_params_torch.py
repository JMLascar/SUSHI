"""
PyTorch port of SUSHI_PCANN_linked_params.py

Key changes from the JAX version:
- jnp.* → torch.*
- arr.at[i].set(v) → arr = arr.clone(); arr[i] = v
- jax.grad(f)(x) → x.requires_grad_(True); f(x).backward(); g = x.grad
- jax.jit dropped (PyTorch is eager; use torch.compile optionally)
- Starlet_Forward2D → from pyStarlet_2D1D_torch (numpy-backed, bit-identical)
- NN model(Theta) → differentiable nn.Sequential forward pass
"""

import copy
import pickle
import warnings

import numpy as np
import torch
from tqdm import trange

import sys, os
sys.path.insert(0, 'code/')
from training_PCA_NN import untransform_physpar   # pure numpy — unchanged
from pyStarlet_2D1D_torch import Starlet_Forward2D


#####################################
# SUSHI
# Credit: Julia Lascar 2023 / Jax acceleration 2024 / Linked parameters: 2025
# PyTorch port 2026
#####################################

def SUSHI(X_im, *models_dir,
          component_names=("Thermal", "Synchrotron"),
          niter=10000, stop=1e-6, J=2, kmad=1,
          background=None,
          Cost_function="Poisson",
          Chi_2_sigma=None, Custom_Cost=None,
          alpha_A=None, alpha_T=None,
          intermediate_save=False,
          file_name="Sushi_result",
          save=100, restart=None,
          device=None):
    """
    Semi-blind Unmixing with Sparsity for Hyperspectral Images (PyTorch version).

    INPUT — same as JAX version, plus:
    device : torch.device to run on (default: auto-select mps > cuda > cpu)

    OUTPUT — same dictionary as the JAX version:
        "Theta"     : best-fit spectral parameters (numpy arrays)
        "Amplitude" : brightness map per component (numpy array)
        "XRec"      : reconstructed cubes per component + "Total" (numpy)
        "Likelihood": list of cost values per iteration
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"SUSHI running on device: {device}")

    # ───────────────────────── SET UP ─────────────────────────
    l, m, n = X_im.shape
    L, M, N = l, m, n
    print(f"Shape of the data: {l} channels, {m}x{n} pixels.")

    mask_amp = 1  # ignore wavelet coefficients on zero-count pixels

    # Vectorized image  (m*n, l)  — kept as torch tensor on device
    X_np = np.transpose(X_im, (1, 2, 0)).reshape(m * n, l)
    X_vec = torch.tensor(X_np, dtype=torch.float32, device=device)

    class Component:
        def __init__(self, name, model, param_names, first_param, params_cnst):
            self.name = name
            self.model = model        # callable nn.Sequential or equivalent
            self.N_P = len(param_names)
            self.first_param = first_param
            self.param_names = param_names
            self.params_cnst = params_cnst
            self.linked = []

    Component_list = []
    print("Models:")
    for i, mod in enumerate(models_dir):
        print(component_names[i])
        Component_list.append(Component(component_names[i], mod["model"],
                                        mod["param_names"], mod["first_param"],
                                        mod["params_cnst"]))
    N_C = len(Component_list)
    print(f"Number of components: {N_C}")

    # ─── Determine linked parameters ───────────────────────────
    Param_names = [c.param_names for c in Component_list]

    if N_C == 2:
        Sets = [set(p) for p in Param_names]
        linked_params = set.intersection(*Sets)
    elif N_C > 2:
        linked_params = set()
        for i in range(N_C - 1):
            linked_params = linked_params.union(
                set.intersection(set(Param_names[i]), set(Param_names[i + 1])))
    else:
        linked_params = set()

    Link_Count = np.zeros(len(linked_params))
    Parent_Model_List = np.zeros(len(linked_params), dtype=int)
    Parent_Param_List = np.zeros(len(linked_params), dtype=int)

    for c in range(N_C):
        Component_list[c].linked = []
        for i, p in enumerate(Param_names[c]):
            for j, s in enumerate(linked_params):
                if s == p:
                    if Link_Count[j] == 0:
                        Parent_Model_List[j] = c
                        Parent_Param_List[j] = i
                        Link_Count[j] = 1
                        Component_list[c].linked.append({
                            "param_name": p, "param_index": i,
                            "parent_model_index": int(Parent_Model_List[j]),
                            "parent_param_index": int(Parent_Param_List[j])})
                    else:
                        Component_list[c].linked.append({
                            "param_name": p, "param_index": i,
                            "parent_model_index": int(Parent_Model_List[j]),
                            "parent_param_index": int(Parent_Param_List[j])})

    NP_total = 0
    NP_array = np.zeros(N_C, dtype=int)
    NP_arraystart = np.zeros(N_C, dtype=int)
    for c in range(N_C):
        NP_total += Component_list[c].N_P
        NP_array[c] = Component_list[c].N_P
    for c in range(N_C - 1):
        NP_arraystart[c + 1] = NP_arraystart[c] + NP_array[c]

    for c in range(N_C):
        Comp = Component_list[c]
        for s in Comp.linked:
            if s["parent_model_index"] != c:
                Child_Model = component_names[c]
                Child_Param = Comp.param_names[s["param_index"]]
                Parent_Model = component_names[s["parent_model_index"]]
                Parent_Param = Component_list[s["parent_model_index"]].param_names[s["parent_param_index"]]
                print(f"Parameter {Child_Param} in model {Child_Model} is linked to"
                      f" Parameter {Parent_Param} in model {Parent_Model}")
            else:
                print(f"Parameter {Comp.param_names[s['param_index']]} marked as linked.")
                print(f"Model {component_names[c]} marked as parent model.")

    # ─── Background ────────────────────────────────────────────
    if background is not None:
        if background.ndim == 1:
            bg_np = np.zeros((m * n, l))
            bg_np[:, :] = background[np.newaxis, :]
        else:
            bg_np = background.reshape(l, m * n).T
        bg_vec = torch.tensor(bg_np, dtype=torch.float32, device=device)
    else:
        bg_vec = torch.zeros(1, device=device)  # broadcasts

    # ─── Helper: slice per-component parameters ─────────────────
    def theta_c(Theta_all, c):
        return Theta_all[:, NP_arraystart[c]: NP_arraystart[c] + NP_array[c]]

    # ─── Link parameters (apply constraints) ───────────────────
    def Link_Params(Theta):
        # Theta: (m*n, NP_total) torch tensor
        Theta = Theta.clone()
        for index in range(N_C):
            Comp = Component_list[index]
            for s in Comp.linked:
                if s["parent_model_index"] != index:
                    parent_theta = theta_c(Theta, s["parent_model_index"])
                    new_val = parent_theta[:, s["parent_param_index"]]
                    col = NP_arraystart[index] + s["param_index"]
                    Theta[:, col] = new_val
        return Theta

    # ─── Spectral reconstruction ────────────────────────────────
    def X_Recover(Theta_all, Amp_all):
        """
        Theta_all : (m*n, NP_total) -- must carry gradients if backward is needed
        Amp_all   : (N_C, m*n)
        Returns dict of (m*n, l) tensors.
        """
        XRec = {"Total": bg_vec.expand(m * n, l).clone()}
        Theta_all = Link_Params(Theta_all)
        for index, mod_name in enumerate(component_names):
            Theta = theta_c(Theta_all, index)
            # model: (m*n, N_P) -> (m*n, rank_PCA)
            # then untransform_spec maps PCA space → spectral space
            spec = Component_list[index].model(Theta)  # (m*n, l) already in phys units
            amp = Amp_all[index, :].unsqueeze(1)       # (m*n, 1)
            comp = torch.real(spec) * amp
            XRec[mod_name] = comp
            XRec["Total"] = XRec["Total"] + comp
        return XRec

    # ─── Cost function ──────────────────────────────────────────
    def get_cost(Theta_all, Amp_all):
        XRec = X_Recover(Theta_all, Amp_all)
        total = XRec["Total"]
        Mask = (total > 0).float()
        if Cost_function == "Poisson":
            cost = (total * Mask - X_vec * Mask * torch.log(total.abs() + 1e-14)).sum()
        elif Cost_function == "Chi_2":
            if Chi_2_sigma is not None:
                sig = torch.tensor(Chi_2_sigma, dtype=torch.float32, device=device)
                cost = (((X_vec * Mask - total * Mask) ** 2) / sig ** 2).sum()
            else:
                cost = ((X_vec * Mask - total * Mask) ** 2).sum()
        elif Cost_function == "Custom":
            cost = Custom_Cost(X_vec, total)
        else:
            raise ValueError("Cost_function must be Poisson, Chi_2, or Custom.")
        return cost

    # ─── Gradient helpers (auto-diff through the NN) ────────────
    def compute_grad_theta(Theta, Amp):
        """Returns ∂cost/∂Theta without updating Theta in place."""
        T = Theta.detach().requires_grad_(True)
        A = Amp.detach()
        loss = get_cost(T, A)
        loss.backward()
        return T.grad.detach()

    def compute_grad_amp(Theta, Amp):
        """Returns ∂cost/∂Amp without updating Amp in place."""
        T = Theta.detach()
        A = Amp.detach().requires_grad_(True)
        loss = get_cost(T, A)
        loss.backward()
        return A.grad.detach()

    # ─── Regularization ────────────────────────────────────────
    def mad(z):
        return torch.median(torch.abs(z - torch.median(z))) / 0.6735

    def reg_grad_thrs(Lambda, grad_lambda, alpha_T_val, Amp, NA):
        """
        Wavelet-domain sparsity regularization for parameter maps.
        Lambda   : (m*n, NA)  torch tensor
        grad_lambda : (m*n, NA) torch tensor
        Returns regularized (m*n, NA) tensor.
        """
        Lambda_map = Lambda.reshape(M, N, NA)
        grad_map   = grad_lambda.reshape(M, N, NA)
        Output = Lambda_map.clone()

        Amp_map = Amp.reshape(M, N)

        for i in range(NA):
            x        = Lambda_map[:, :, i]
            g        = grad_map[:, :, i]

            # 2D Starlet forward — uses numpy internally then converts to torch
            x_np = x.cpu().numpy()
            g_np = g.cpu().numpy()
            c,  w  = Starlet_Forward2D(torch.from_numpy(x_np), J=J, M=M, N=N)
            cg, wg = Starlet_Forward2D(torch.from_numpy(g_np), J=J, M=M, N=N)

            w = w.clone()
            for r in range(J):
                wg_mask = wg[r][Amp_map.cpu() > mask_amp]
                thrd = kmad * alpha_T_val * mad(wg_mask)
                wr = w[r]
                w[r] = (wr - thrd * wr.sign()) * (wr.abs() > thrd)

            rec = c + w.sum(dim=0)   # sum all planes
            Output[:, :, i] = rec

        return Output.reshape(M * N, NA)

    # ─── First guess ────────────────────────────────────────────
    Amp_all = torch.zeros(N_C, m * n, device=device)
    for c in range(N_C):
        Amp_all[c] = X_vec.sum(dim=1) / N_C

    Theta_all = torch.zeros(m * n, NP_total, device=device)
    for c in range(N_C):
        for i in range(Component_list[c].N_P):
            col = NP_arraystart[c] + i
            Theta_all[:, col] = Component_list[c].first_param[i]

    # ─── Gradient step sizes ────────────────────────────────────
    if alpha_T is None:
        alpha_T_val = 0.1 / float(X_vec.sum(dim=1).max().item())
    else:
        alpha_T_val = float(alpha_T)
    if alpha_A is None:
        alpha_A_val = 1.0
    else:
        alpha_A_val = float(alpha_A)

    # ─── Restart from previous result ───────────────────────────
    Acc = []
    if restart is not None:
        Theta_all = torch.tensor(
            restart["Theta"].reshape(m * n, NP_total), dtype=torch.float32, device=device)
        Amp_all = torch.tensor(
            restart["Amplitude"].reshape(m * n, N_C).T, dtype=torch.float32, device=device)
        Acc = list(restart["Likelihood"])

    # ─── MAIN LOOP ──────────────────────────────────────────────
    t = trange(niter, desc='Loss', leave=True)
    for i in t:
        # ── Descent on Theta ──
        Theta_grad = compute_grad_theta(Theta_all, Amp_all)
        Theta_gd = Theta_all.detach() - alpha_T_val * Theta_grad

        new_Theta = Theta_all.detach().clone()
        for c in range(N_C):
            start, end = int(NP_arraystart[c]), int(NP_arraystart[c] + NP_array[c])
            Lamb  = Theta_gd[:, start:end]
            Grads = Theta_gd[:, start:end]   # wavelet reg re-uses gradient-descended values
            Theta_reg = reg_grad_thrs(Lamb, Grads,
                                      alpha_T_val=alpha_T_val,
                                      Amp=Amp_all[c].detach(),
                                      NA=int(NP_array[c]))
            new_Theta[:, start:end] = Theta_reg.to(device)
        Theta_all = Link_Params(new_Theta)

        # ── Descent on Amplitude ──
        Amp_grad = compute_grad_amp(Theta_all, Amp_all)
        new_Amp = Amp_all.detach() - alpha_A_val * Amp_grad
        new_Amp = new_Amp.clamp(min=0)
        new_Amp[new_Amp == 0] = 1e-14
        Amp_all = new_Amp

        # ── Cost ──────────────────────────────────────────────────
        with torch.no_grad():
            likelihood = get_cost(Theta_all, Amp_all).item()
        Acc.append(likelihood)

        mean_diff = 0.0
        if i > 150:
            A1 = np.array(Acc[-150:-100])
            A2 = np.array(Acc[-50:])
            mean_diff = float(np.mean(A2 - A1) / (np.mean(A1) + 1e-30))
            if mean_diff < stop:
                print("Stopping criterion reached.")
                break

        t.set_description(f"Loss={likelihood:.4e}, Mean diff={mean_diff:.2e}")
        t.refresh()

        if intermediate_save and i % save == 0:
            _save_results(file_name, i, Theta_all, Amp_all, component_names,
                          Component_list, NP_arraystart, NP_array,
                          X_Recover, Acc, m, n, l)

    # ─── Final results ──────────────────────────────────────────
    with torch.no_grad():
        XRec_t = X_Recover(Theta_all, Amp_all)

    Theta_res = []
    for c in range(N_C):
        T_np = theta_c(Theta_all, c).detach().cpu().numpy()
        T_phys = untransform_physpar(T_np, Component_list[c].params_cnst)
        Theta_res.append(T_phys.reshape(m, n, int(NP_array[c])))

    Results = {
        "Theta":     Theta_res,
        "Amplitude": Amp_all.T.detach().cpu().numpy().reshape(m, n, N_C),
        "XRec":      {},
        "Likelihood": Acc,
    }
    Results["XRec"]["Total"] = XRec_t["Total"].detach().cpu().numpy().T.reshape(l, m, n)
    for mod_name in component_names:
        Results["XRec"][mod_name] = XRec_t[mod_name].detach().cpu().numpy().T.reshape(l, m, n)

    if np.isnan(Acc[-1]):
        print("NaN in likelihood — try lowering alpha_T / alpha_A.")

    with open(f"{file_name}.p", "wb") as f:
        pickle.dump(Results, f)

    return Results


def _save_results(file_name, i, Theta_all, Amp_all, component_names,
                  Component_list, NP_arraystart, NP_array, X_Recover, Acc, m, n, l):
    XRec_t = X_Recover(Theta_all, Amp_all)
    Results = {
        "Theta":      Theta_all.detach().cpu().numpy(),
        "Amplitude":  Amp_all.detach().cpu().numpy(),
        "Likelihood": Acc,
        "XRec":       {k: v.detach().cpu().numpy() for k, v in XRec_t.items()},
    }
    with open(f"{file_name}_{i}.p", "wb") as f:
        pickle.dump(Results, f)
