"""Receive a UDP transport stream, demux the KLV and print UAV telemetry.

    python recv_udp.py --port 5600
    python recv_udp.py --port 5601 --save capture.ts --json telemetry.jsonl
    python recv_udp.py --port 5600 --group 239.1.1.1 --iface 192.168.1.127
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import struct
import sys
import time

from uavstream.klv import checksum_ok, decode_to_dict
from uavstream.tsparse import LiveDemux

HERE = pathlib.Path(__file__).parent

# what gets printed on the one-line summary, in order
SUMMARY_KEYS = [
    ("Sensor Latitude", "lat", "{:.6f}"),
    ("Sensor Longitude", "lon", "{:.6f}"),
    ("Sensor True Altitude", "alt", "{:.0f}m"),
    ("Platform Heading Angle", "hdg", "{:.1f}deg"),
    ("Frame Center Latitude", "fc_lat", "{:.6f}"),
    ("Frame Center Longitude", "fc_lon", "{:.6f}"),
    ("Slant Range", "range", "{:.0f}m"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0", help="local address to bind")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--group", help="multicast group to join")
    ap.add_argument("--iface", default="0.0.0.0", help="local IP for the multicast join")
    ap.add_argument("--save", help="write the received TS to this file")
    ap.add_argument("--json", dest="json_out", help="append decoded telemetry as JSON lines")
    ap.add_argument("--full", action="store_true", help="print every KLV tag, not a summary")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="exit after this many seconds with no data (0 = never)")
    ap.add_argument("--count", type=int, default=0, help="stop after N KLV packets")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
    sock.bind((args.bind, args.port))
    if args.group:
        mreq = struct.pack("4s4s", socket.inet_aton(args.group), socket.inet_aton(args.iface))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(1.0)

    ts_file = open(HERE / args.save, "wb") if args.save else None
    json_file = open(HERE / args.json_out, "a", encoding="utf-8") if args.json_out else None

    demux = LiveDemux()
    dgrams = total_bytes = klv_count = bad_checksum = 0
    start = time.perf_counter()
    last_data = start
    last_report = start

    where = f"{args.group} via {args.bind}" if args.group else args.bind
    print(f"listening on udp://{where}:{args.port}  Ctrl+C to stop\n")

    try:
        while True:
            try:
                payload, addr = sock.recvfrom(65535)
            except socket.timeout:
                if args.timeout and time.perf_counter() - last_data > args.timeout:
                    print("timed out waiting for data")
                    break
                continue

            now = time.perf_counter()
            last_data = now
            dgrams += 1
            total_bytes += len(payload)
            if ts_file:
                ts_file.write(payload)

            for pid, klv in demux.feed(payload):
                klv_count += 1
                ok = checksum_ok(klv)
                if not ok:
                    bad_checksum += 1
                values = decode_to_dict(klv)

                if json_file:
                    json_file.write(json.dumps(
                        {"recv_time": time.time(), "src": f"{addr[0]}:{addr[1]}",
                         "pid": pid, "checksum_ok": ok, **values}) + "\n")

                if args.full:
                    print(f"--- KLV #{klv_count} pid 0x{pid:04X} "
                          f"{len(klv)} bytes checksum={'OK' if ok else 'BAD'}")
                    for key, value in values.items():
                        print(f"    {key:<28} {value}")
                else:
                    parts = []
                    for key, label, fmt in SUMMARY_KEYS:
                        value = values.get(key)
                        if isinstance(value, (int, float)):
                            parts.append(f"{label}={fmt.format(value)}")
                    flag = "" if ok else "  [CHECKSUM BAD]"
                    print(f"#{klv_count:<5} 0x{pid:04X}  " + "  ".join(parts) + flag)

                if args.count and klv_count >= args.count:
                    raise KeyboardInterrupt

            if now - last_report >= 5.0:
                elapsed = now - start
                print(f"  .. {dgrams:,} datagrams  {total_bytes / 1e6:.2f} MB  "
                      f"{total_bytes * 8 / elapsed / 1e6:.2f} Mbps  "
                      f"{klv_count:,} KLV packets  from {addr[0]}")
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if ts_file:
            ts_file.close()
        if json_file:
            json_file.close()
        elapsed = max(time.perf_counter() - start, 1e-9)
        print(f"\nreceived {dgrams:,} datagrams / {total_bytes / 1e6:.2f} MB "
              f"({total_bytes * 8 / elapsed / 1e6:.2f} Mbps)")
        print(f"KLV packets: {klv_count:,}   bad checksums: {bad_checksum}")
        if demux.streams:
            pids = ", ".join(f"0x{p:04X} (type 0x{t:02X})" for p, t in sorted(demux.streams.items()))
            print(f"elementary streams seen: {pids}")
        if args.save:
            print(f"TS written to {HERE / args.save}")
        if args.json_out:
            print(f"telemetry written to {HERE / args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
