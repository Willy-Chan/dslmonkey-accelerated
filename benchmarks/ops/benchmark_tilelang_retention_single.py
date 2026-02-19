import argparse
import importlib.util

import torch
import triton


def _load_tilelang_retention_impl():
    spec_tilelang = importlib.util.spec_from_file_location(
        "tilelang_rebased",
        "/home/simon/willyc/dslmonkey-accelerated/REBASED_SOLUTIONS/lv5-s3-k5.py",
    )
    module_tilelang = importlib.util.module_from_spec(spec_tilelang)
    spec_tilelang.loader.exec_module(module_tilelang)
    return module_tilelang.ModelNew


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--single-run", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = "cuda"
    dtype = torch.float16

    torch.manual_seed(0)
    q = torch.randn(args.B, args.H, args.T, args.D, device=device, dtype=dtype)
    k = torch.randn(args.B, args.H, args.T, args.D, device=device, dtype=dtype)
    v = torch.randn(args.B, args.H, args.T, args.D, device=device, dtype=dtype)

    TileLangRetentionParallelImpl = _load_tilelang_retention_impl()
    model = TileLangRetentionParallelImpl().to(device)

    torch.set_grad_enabled(False)

    for _ in range(args.warmup):
        model(q, k, v)
    torch.cuda.synchronize()

    if args.single_run:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(q, k, v)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        print(f"TileLangRetentionParallelImpl latency (single run), T={args.T}: {ms:.6f} ms")
        return

    quantiles = [0.5, 0.2, 0.8]
    med_ms, p20_ms, p80_ms = triton.testing.do_bench(lambda: model(q, k, v), quantiles=quantiles)
    print(f"TileLangRetentionParallelImpl latency, T={args.T}: {med_ms:.6f} ms (p20={p20_ms:.6f}, p80={p80_ms:.6f})")


if __name__ == "__main__":
    main()
