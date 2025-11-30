import torch as t


def q_aware_linear_forward(x_meta, w_meta, L_w, S_w) -> t.Tensor:

    # x_q: (B, FIN)
    x_q, x_scale, _ = x_meta.data, x_meta.step_size, x_meta.zero_point

    # w_q: (FOUT * L_w, FIN)
    w_q, w_scale, _ = w_meta.data, w_meta.step_size, w_meta.zero_point

    B = x_q.shape[0]
    FOUT = w_q.size(0) // L_w

    y_q = t.matmul(x_q, w_q.T)  # shape: (B, FOUT * L_w)

    slice_shifts = t.arange(L_w, device=w_q.device, dtype=t.int64) * S_w  # (L_w,)
    shift_vec = t.tile(slice_shifts, (FOUT,))  # (FOUT*L_w,)
    pow2 = (2.0**shift_vec).to(y_q.dtype)  # (FOUT*L_w,)

    x_scale_b = (
        x_scale
        if isinstance(x_scale, t.Tensor)
        else t.tensor(x_scale, device=y_q.device, dtype=y_q.dtype)
    )
    x_scale_b = (
        x_scale_b.reshape(-1, 1) if x_scale_b.ndim <= 1 else x_scale_b
    )  # (B,1) or (B,1,...)

    # w_scale: () or (FOUT*L_w,) -> (1, FOUT*L_w)
    w_scale_b = (
        w_scale
        if isinstance(w_scale, t.Tensor)
        else t.tensor(w_scale, device=y_q.device, dtype=y_q.dtype)
    )
    w_scale_b = w_scale_b.reshape(1, -1)

    c_q = x_scale_b * (w_scale_b * pow2.reshape(1, -1))  # (B, FOUT*L_w)

    y = (y_q * c_q).reshape(B, FOUT, L_w).sum(dim=2)

    return y
