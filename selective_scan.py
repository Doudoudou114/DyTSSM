import torch
import torch.nn.functional as F


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    source_dtype = u.dtype
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()

    batch_size, channels, length = u.shape
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)

    scan_count = B.shape[1]
    channels_per_scan = channels // scan_count
    state_size = A.shape[-1]

    u = u.view(batch_size, scan_count, channels_per_scan, length)
    delta = delta.view(batch_size, scan_count, channels_per_scan, length)
    A = A.view(scan_count, channels_per_scan, state_size)
    B = B.view(batch_size, scan_count, state_size, length)
    C = C.view(batch_size, scan_count, state_size, length)

    state = u.new_zeros(batch_size, scan_count, channels_per_scan, state_size)
    outputs = []
    for index in range(length):
        step_delta = delta[..., index].unsqueeze(-1)
        decay = torch.exp(step_delta * A.unsqueeze(0))
        update = step_delta * B[..., index].unsqueeze(2)
        state = decay * state + update * u[..., index].unsqueeze(-1)
        outputs.append((state * C[..., index].unsqueeze(2)).sum(dim=-1))

    output = torch.stack(outputs, dim=-1).reshape(batch_size, channels, length)
    if D is not None:
        output = output + D.view(1, -1, 1) * u.reshape(batch_size, channels, length)
    return output.to(source_dtype)
