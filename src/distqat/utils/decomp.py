import torch as t


def _twos_complement_decode(ui: t.Tensor, bits: int) -> t.Tensor:
    """
    Decode an unsigned bit pattern in [0, 2^bits) to a signed two's-complement tensor.
    Used to interpret the designated signed slice in slice-wise quantization so that
    decompose/recompose remain bit-true under the FPG.

    Parameters
    ----------
    ui : t.Tensor
        Tensor of non-negative integers representing 2's-complement bit patterns.
    bits : int
        Bit-width of the representation.

    Returns
    -------
    t.Tensor
        Signed integer tensor (int32) with the same shape as `ui`.
    """
    sign = 1 << (bits - 1)
    mod = 1 << bits
    ui32 = ui.to(t.int32)
    return t.where((ui32 & sign) != 0, ui32 - mod, ui32)


def _twos_complement_encode(s: t.Tensor, bits: int) -> t.Tensor:
    """
    Encode a signed S-bit integer tensor as its unsigned two's-complement bit pattern
    in [0, 2^bits). Used during recomposition when treating the signed slice as an
    unsigned digit.

    Parameters
    ----------
    s : t.Tensor
        Signed integer tensor to encode.
    bits : int
        Bit-width to encode into.

    Returns
    -------
    t.Tensor
        Unsigned integer tensor (int64) in [0, 2^bits) with the same shape as `s`.
    """
    mod = 1 << bits
    return t.remainder(s.to(t.int64), mod).to(t.int64)


def _recomposition_check(slices_msb: t.Tensor, w_q_full: t.Tensor, L_w: int, S_w: int):
    """
    Bit-true recomposition from centered slices under FPG.
    Interprets the first slice as signed two's-complement and recenters remaining
    slices by +2^(S_w-1), then shifts/accumulates and converts back to a signed
    integer. Asserts exact equality with the original integer weights.

    Parameters
    ----------
    slices_msb : t.Tensor
        Weight slices of shape (L_w, F_out, F_in), int32. First slice is signed; others are centered.
    w_q_full : t.Tensor
        Reference integer weights of shape (F_out, F_in), typically int32.
    L_w : int
        Number of slices (must equal `slices_msb.shape[0]`).
    S_w : int
        Bit-width per slice.

    Raises
    ------
    AssertionError
        If recomposition does not exactly match `w_q_full`.
    """
    total_bits = L_w * S_w
    L, f_out, f_in = slices_msb.shape
    assert L == L_w
    recon = t.zeros(f_out, f_in, dtype=t.int64, device=slices_msb.device)

    for k in range(L_w):
        shift = total_bits - (k + 1) * S_w
        if k == 0:
            digit_u = _twos_complement_encode(slices_msb[k], S_w)
        else:
            digit_u = slices_msb[k].to(t.int64) + (1 << (S_w - 1))
        recon += digit_u << shift

    signbit = 1 << (total_bits - 1)
    modN = 1 << total_bits
    recon_signed = t.where((recon & signbit) != 0, recon - modN, recon).to(t.int32)

    assert t.equal(
        recon_signed, w_q_full
    ), "Slice recomposition mismatch (MSB-first centered)."


def decompose(int_weight: t.Tensor, L_w: int, S_w: int) -> t.Tensor:
    """
    Decompose signed integer weights into `L_w` slices of width `S_w`.
    The first slice is decoded as signed two's-complement; remaining slices are
    centered via (u - 2^(S_w-1)).

    Parameters
    ----------
    int_weight : t.Tensor
        Signed integer weight tensor, dtype in {int16, int32, int64}.
    L_w : int
        Number of slices to produce.
    S_w : int
        Bit-width per slice.

    Returns
    -------
    t.Tensor
        Stacked slices of shape (L_w, F_out, F_in), dtype int32.
    """
    assert int_weight.dtype in (t.int16, t.int32, t.int64)

    mask = (1 << S_w) - 1
    total_bits = L_w * S_w

    out = []
    for k in range(L_w):
        # Bits for slice k (MSB-first):
        #   [total_bits-(k+1)S_w, total_bits-k*S_w)
        shift = total_bits - (k + 1) * S_w
        ui = ((int_weight >> shift) & mask).to(t.int32)
        if k == 0:
            si = _twos_complement_decode(ui, S_w)  # signed MSB slice
        else:
            si = ui - (1 << (S_w - 1))  # centered lower slice
        out.append(si)

    return t.stack(out, dim=0)  # (L_w, f_out, f_in)


if __name__ == "__main__":
    w_q_full = t.randint(-100, 100, (10, 10), dtype=t.int32)
    L_w = 2
    S_w = 4
    slices_msb = decompose(w_q_full, L_w, S_w)

    _recomposition_check(slices_msb, w_q_full, L_w, S_w)
