import argparse
import socket
import subprocess


def run_command(command):
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )


def resolve_peer(peer_host):
    return socket.gethostbyname(peer_host)


def interface_for_peer(peer_host):
    peer_ip = resolve_peer(peer_host)

    result = run_command(
        ["ip", "route", "get", peer_ip]
    )

    tokens = result.stdout.split()

    if "dev" not in tokens:
        raise RuntimeError(
            f"Could not determine interface for peer "
            f"{peer_host} ({peer_ip}). "
            f"Route output: {result.stdout.strip()}"
        )

    dev_index = tokens.index("dev") + 1

    if dev_index >= len(tokens):
        raise RuntimeError(
            f"Invalid route output for {peer_host}: "
            f"{result.stdout.strip()}"
        )

    interface = tokens[dev_index]

    return peer_ip, interface


def configure_netem(
    peer_host,
    rate_mbps,
    delay_ms=0.0,
    queue_limit=10000,
):
    peer_ip, interface = interface_for_peer(peer_host)

    command = [
        "tc",
        "qdisc",
        "replace",
        "dev",
        interface,
        "root",
        "netem",
    ]

    if delay_ms > 0:
        command.extend(
            [
                "delay",
                f"{delay_ms}ms",
            ]
        )

    command.extend(
        [
            "rate",
            f"{rate_mbps}mbit",
            "limit",
            str(queue_limit),
        ]
    )

    run_command(command)

    status = run_command(
        [
            "tc",
            "qdisc",
            "show",
            "dev",
            interface,
        ]
    ).stdout.strip()

    return {
        "peer_host": peer_host,
        "peer_ip": peer_ip,
        "interface": interface,
        "rate_mbps": rate_mbps,
        "delay_ms": delay_ms,
        "queue_limit": queue_limit,
        "status": status,
    }


def clear_netem(peer_host):
    peer_ip, interface = interface_for_peer(peer_host)

    subprocess.run(
        [
            "tc",
            "qdisc",
            "del",
            "dev",
            interface,
            "root",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    return peer_ip, interface


def main():
    parser = argparse.ArgumentParser(
        description="Configure UCC network emulation."
    )

    parser.add_argument(
        "--peer",
        required=True,
        help="Peer hostname used to resolve the output interface.",
    )

    parser.add_argument(
        "--rate-mbps",
        type=float,
        required=True,
        help="Aggregate interface rate in Mbit/s.",
    )

    parser.add_argument(
        "--delay-ms",
        type=float,
        default=0.0,
        help="Fixed one-way delay in milliseconds.",
    )

    args = parser.parse_args()

    result = configure_netem(
        peer_host=args.peer,
        rate_mbps=args.rate_mbps,
        delay_ms=args.delay_ms,
    )

    print(
        f"[TC] peer={result['peer_host']} "
        f"ip={result['peer_ip']} "
        f"interface={result['interface']}"
    )

    print(
        f"[TC] rate={result['rate_mbps']:.3f} Mbit/s "
        f"delay={result['delay_ms']:.3f} ms "
        f"limit={result['queue_limit']}"
    )

    print(
        f"[TC] {result['status']}"
    )


if __name__ == "__main__":
    main()
