"""Generate a .ts file carrying MISB ST 0601 KLV telemetry for a simulated UAV.

    python make_stream.py --duration 60 --rate 10 --out streams/uav_demo.ts
"""

from __future__ import annotations

import argparse
import pathlib

from uavstream.flight import OrbitPath
from uavstream.tsmux import TsMuxer, TS_PACKET_SIZE


def build(duration: float, rate: float, path: OrbitPath, cbr_bps: int = 0) -> bytes:
    mux = TsMuxer()
    out = bytearray()
    preroll_s = 0.2                      # PTS sits this far ahead of PCR
    frames = int(duration * rate)
    bytes_per_second = cbr_bps / 8 if cbr_bps else 0

    for i in range(frames):
        t = i / rate
        telemetry = path.at(t)
        pcr = int(t * 27_000_000)
        pts = int((t + preroll_s) * 90_000)

        out += mux.pat()
        out += mux.pmt()
        for pkt in mux.klv_pes(telemetry.to_klv(), pts_90k=pts, pcr_27mhz=pcr):
            out += pkt

        if bytes_per_second:  # pad to a constant bitrate so analysers see CBR
            target = int(bytes_per_second * (t + 1 / rate))
            while len(out) + TS_PACKET_SIZE <= target:
                out += mux.null_packet()

    return bytes(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="streams/uav_demo.ts", help="output .ts path")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of telemetry")
    ap.add_argument("--rate", type=float, default=10.0, help="KLV packets per second")
    ap.add_argument("--lat", type=float, default=28.6139, help="orbit centre latitude")
    ap.add_argument("--lon", type=float, default=77.2090, help="orbit centre longitude")
    ap.add_argument("--radius", type=float, default=1500.0, help="orbit radius (m)")
    ap.add_argument("--altitude", type=float, default=1200.0, help="platform altitude (m MSL)")
    ap.add_argument("--speed", type=float, default=45.0, help="ground speed (m/s)")
    ap.add_argument("--target-elev", type=float, default=200.0, help="target elevation (m)")
    ap.add_argument("--cbr", type=int, default=0,
                    help="pad with null packets to this constant bitrate (bps), 0 = off")
    args = ap.parse_args()

    path = OrbitPath(args.lat, args.lon, args.radius, args.altitude,
                     args.speed, args.target_elev)
    data = build(args.duration, args.rate, path, args.cbr)

    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

    packets = len(data) // TS_PACKET_SIZE
    print(f"wrote {dest}  ({len(data):,} bytes, {packets:,} TS packets)")
    print(f"  {args.duration:g}s @ {args.rate:g} KLV/s -> {int(args.duration * args.rate)} metadata frames")
    print(f"  orbit {args.radius:g} m around {args.lat:.5f}, {args.lon:.5f} at {args.altitude:g} m")
    print(f"  average bitrate {len(data) * 8 / args.duration / 1000:.1f} kbps")


if __name__ == "__main__":
    main()
