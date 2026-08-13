import json


def mbit_to_bytes(size_mbit):
    return int(round(size_mbit * 1_000_000 / 8))


def send_json_line(sock, message):
    encoded = (
        json.dumps(message, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")

    sock.sendall(encoded)


def read_json_line(stream):
    line = stream.readline()

    if not line:
        return None

    if isinstance(line, bytes):
        line = line.decode("utf-8")

    return json.loads(line)


def read_exact(stream, num_bytes):
    chunks = []
    remaining = num_bytes

    while remaining > 0:
        chunk = stream.read(remaining)

        if not chunk:
            raise RuntimeError(
                f"Connection closed with "
                f"{remaining} payload bytes still expected."
            )

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def send_frame(sock, metadata, payload):
    metadata = dict(metadata)
    metadata["payload_bytes"] = len(payload)

    send_json_line(sock, metadata)

    if payload:
        sock.sendall(payload)


def read_frame(stream):
    metadata = read_json_line(stream)

    if metadata is None:
        return None, None

    payload_bytes = int(
        metadata.get("payload_bytes", 0)
    )

    payload = read_exact(
        stream,
        payload_bytes
    ) if payload_bytes > 0 else b""

    return metadata, payload
