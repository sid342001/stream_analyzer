"""Stream a .ts file over UDP to one or more destinations (real IP + pseudo IP).

    python stream_udp.py                              # uses config.json
    python stream_udp.py --target 192.168.1.127:5600 --target 127.0.0.1:5601
    python stream_udp.py --file streams/uav_demo.ts --loop

Timing follows the PCR embedded in the stream; if the file has no PCR the
--bitrate value is used instead. Each datagram carries 7 TS packets (1316
bytes) so it fits inside a standard 1500-byte MTU.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import pathlib
import socket
import sys
import time

from uavstream.tsmux import TS_PACKET_SIZE
from uavstream.tsparse import iter_packets, read_pcr

HERE = pathlib.Path(__file__).parent
PCR_WRAP = (1 << 33) * 300  # PCR base is 33 bits at 90 kHz, extension is /300


def load_config() -> dict:
    cfg_path = HERE / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {}


def parse_target(text: str, default_port: int) -> tuple[str, int]:
    if ":" in text:
        host, _, port = text.rpartition(":")
        return host, int(port)
    return text, default_port


def is_multicast(host: str) -> bool:
    try:
        return ipaddress.ip_address(socket.gethostbyname(host)).is_multicast
    except (ValueError, OSError):
        return False


def packet_timeline(data: bytes) -> list[float] | None:
    """Seconds-from-start for every TS packet, interpolated between PCRs."""
    marks: list[tuple[int, int]] = []  # (packet index, unwrapped PCR)
    total = 0
    last_raw = None
    wraps = 0
    for idx, pkt in enumerate(iter_packets(data)):
        total = idx + 1
        pcr = read_pcr(pkt)
        if pcr is None:
            continue
        if last_raw is not None and pcr < last_raw - PCR_WRAP // 2:
            wraps += 1
        last_raw = pcr
        marks.append((idx, pcr + wraps * PCR_WRAP))

    if len(marks) < 2:
        return None

    base = marks[0][1]
    times = [0.0] * total
    for (i0, p0), (i1, p1) in zip(marks, marks[1:]):
        t0, t1 = (p0 - base) / 27e6, (p1 - base) / 27e6
        span = max(i1 - i0, 1)
        for k in range(i0, i1):
            times[k] = t0 + (t1 - t0) * (k - i0) / span
    # extrapolate the head and tail using the nearest known packet rate
    rate = (times[marks[1][0]] - times[marks[0][0]]) / max(marks[1][0] - marks[0][0], 1)
    for k in range(0, marks[0][0]):
        times[k] = times[marks[0][0]] - rate * (marks[0][0] - k)
    last_idx = marks[-1][0]
    times[last_idx] = (marks[-1][1] - base) / 27e6  # never covered by the loop above
    for k in range(last_idx + 1, total):
        times[k] = times[last_idx] + rate * (k - last_idx)
    return times


def precise_sleep(seconds: float) -> None:
    """Windows' sleep granularity is ~15 ms, so spin out the last stretch."""
    if seconds <= 0:
        return
    deadline = time.perf_counter() + seconds
    if seconds > 0.004:
        time.sleep(seconds - 0.003)
    while time.perf_counter() < deadline:
        pass


def make_socket(ttl: int, iface: str | None) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    if iface:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                        socket.inet_aton(iface))
    return sock


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=cfg.get("stream_file", "streams/uav_demo.ts"))
    ap.add_argument("--target", action="append", default=None,
                    help="host:port, repeatable (defaults come from config.json)")
    ap.add_argument("--port", type=int, default=cfg.get("default_port", 5600))
    ap.add_argument("--bitrate", type=int, default=cfg.get("fallback_bitrate", 2_000_000),
                    help="bps, used only when the file carries no PCR")
    ap.add_argument("--packets", type=int, default=7, help="TS packets per datagram")
    ap.add_argument("--ttl", type=int, default=cfg.get("multicast_ttl", 8))
    ap.add_argument("--iface", default=cfg.get("multicast_iface"),
                    help="local IP to send multicast from")
    ap.add_argument("--loop", action="store_true", help="repeat the file forever")
    ap.add_argument("--asap", action="store_true", help="ignore timing, blast at full speed")
    args = ap.parse_args()

    if args.target:
        targets = [parse_target(t, args.port) for t in args.target]
    else:
        targets = [(t["host"], int(t.get("port", args.port)))
                   for t in cfg.get("targets", [{"host": "127.0.0.1"}])]

    path = pathlib.Path(args.file)
    if not path.is_absolute():
        path = HERE / path
    if not path.exists():
        sys.exit(f"stream file not found: {path}\nrun: python make_stream.py")

    data = path.read_bytes()
    if len(data) < TS_PACKET_SIZE:
        sys.exit(f"{path} is too small to be a transport stream")

    times = None if args.asap else packet_timeline(data)
    if times is None and not args.asap:
        n = len(data) // TS_PACKET_SIZE
        per_packet = TS_PACKET_SIZE * 8 / args.bitrate
        times = [i * per_packet for i in range(n)]
        timing = f"CBR {args.bitrate / 1e6:.2f} Mbps (no PCR found)"
    elif args.asap:
        timing = "as fast as possible"
    else:
        timing = "PCR-locked"

    sock = make_socket(args.ttl, args.iface)
    packets = [data[i:i + TS_PACKET_SIZE]
               for i in range(0, len(data) - TS_PACKET_SIZE + 1, TS_PACKET_SIZE)]
    duration = times[-1] if times else 0.0

    print(f"file      : {path.name}  ({len(packets):,} TS packets, {duration:.1f}s)")
    print(f"timing    : {timing}")
    print(f"datagram  : {args.packets * TS_PACKET_SIZE} bytes ({args.packets} TS packets)")
    for host, port in targets:
        kind = "multicast" if is_multicast(host) else "unicast"
        print(f"target    : udp://{host}:{port}  [{kind}]")
    print("streaming... Ctrl+C to stop\n")

    sent_bytes = 0
    sent_dgrams = 0
    lap = 0
    start = time.perf_counter()
    next_report = start + 1.0

    try:
        while True:
            epoch = time.perf_counter()
            for i in range(0, len(packets), args.packets):
                chunk = packets[i:i + args.packets]
                payload = b"".join(chunk)

                if times is not None:
                    precise_sleep((epoch + times[i]) - time.perf_counter())

                for host, port in targets:
                    try:
                        sock.sendto(payload, (host, port))
                    except OSError as exc:
                        print(f"  ! {host}:{port} {exc}")
                sent_bytes += len(payload) * len(targets)
                sent_dgrams += len(targets)

                now = time.perf_counter()
                if now >= next_report:
                    elapsed = now - start
                    print(f"  [{elapsed:7.1f}s] {sent_dgrams:>8,} datagrams  "
                          f"{sent_bytes / 1e6:>8.2f} MB  "
                          f"{sent_bytes * 8 / elapsed / 1e6:>6.2f} Mbps total")
                    next_report = now + 1.0

            lap += 1
            if not args.loop:
                break
            print(f"  -- loop {lap} complete, restarting --")
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        sock.close()
        elapsed = time.perf_counter() - start
        print(f"sent {sent_dgrams:,} datagrams / {sent_bytes / 1e6:.2f} MB in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
