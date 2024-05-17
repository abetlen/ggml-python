import ggml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--free_mem", type=int, default=1 << 30)
    parser.add_argument("--total_mem", type=int, default=1 << 30)
    parser.add_argument("--backend", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Starting worker on {args.host}:{args.port}")
    print(f"Free memory: {args.free_mem} bytes")
    print(f"Total memory: {args.total_mem} bytes")
    print(f"Backend: {args.backend}")

    if args.backend == "cpu":
        backend = ggml.ggml_backend_cpu_init()
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    assert backend is not None, "Failed to initialize CPU backend"

    endpoints = "{}:{}".format(args.host, args.port).encode("utf-8")

    free_mem = args.free_mem
    total_mem = args.total_mem

    ggml.start_rpc_server(backend, endpoints, free_mem, total_mem)


if __name__ == "__main__":
    main()