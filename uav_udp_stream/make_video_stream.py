"""Build a .ts carrying H.264 video AND MISB ST 0601 KLV telemetry.

ffmpeg renders a synthetic UAV-style video (crosshair + HUD); this script then
folds the KLV metadata in on its own PID, synchronised to the video clock.

    python make_video_stream.py --duration 30 --fps 30 --klv-rate 10
    python make_video_stream.py --source "path/to/real_uav.mp4"

The result plays in VLC/ffplay as normal video, and recv_udp.py decodes the
telemetry from the same stream.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile

from uavstream import ffmpeg
from uavstream.flight import OrbitPath
from uavstream.remux import remux_with_klv
from uavstream.tsmux import TS_PACKET_SIZE

HERE = pathlib.Path(__file__).parent
FONT = pathlib.Path("C:/Windows/Fonts/consola.ttf")


def build_filter(width: int, height: int, hud: bool) -> str:
    """A white crosshair, plus an optional burnt-in HUD showing frame/time."""
    parts = [
        f"drawbox=x=iw/2-1:y=0:w=2:h=ih:color=white@0.6:t=fill",
        f"drawbox=x=0:y=ih/2-1:w=iw:h=2:color=white@0.6:t=fill",
        # corner ticks around the centre reticle
        f"drawbox=x=iw/2-40:y=ih/2-40:w=80:h=80:color=lime@0.9:t=2",
    ]
    if hud and FONT.exists():
        font = str(FONT).replace("\\", "/").replace(":", "\\:")
        parts.append(
            f"drawtext=fontfile='{font}':text='UAV-01 EO  FRAME %{{n}}  T %{{pts}}s'"
            f":x=24:y=24:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.5:boxborderw=8"
        )
        parts.append(
            f"drawtext=fontfile='{font}':text='LAT/LON IN KLV METADATA'"
            f":x=24:y=h-48:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=6"
        )
    return ",".join(parts)


def render_video(ff: str, out: pathlib.Path, duration: float, fps: int,
                 width: int, height: int, bitrate: str, source: str | None,
                 hud: bool) -> None:
    if source:
        inp = ["-stream_loop", "-1", "-i", source, "-t", str(duration)]
    else:
        inp = ["-f", "lavfi", "-i",
               f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}"]

    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", *inp,
           "-vf", build_filter(width, height, hud),
           "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
           "-pix_fmt", "yuv420p", "-b:v", bitrate, "-maxrate", bitrate,
           "-bufsize", bitrate, "-g", str(fps), "-keyint_min", str(fps),
           "-an", "-muxrate", "0", "-f", "mpegts", str(out)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and hud:  # HUD filter can fail on odd font setups
        print("  (HUD render failed, retrying with crosshair only)")
        return render_video(ff, out, duration, fps, width, height, bitrate,
                            source, hud=False)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{result.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="streams/uav_video.ts")
    ap.add_argument("--source", help="use a real video file instead of the synthetic feed")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--bitrate", default="4M", help="video bitrate, e.g. 2M / 4M / 8M")
    ap.add_argument("--klv-rate", type=float, default=10.0, help="KLV packets per second")
    ap.add_argument("--no-hud", action="store_true", help="skip the burnt-in HUD text")
    ap.add_argument("--lat", type=float, default=28.6139)
    ap.add_argument("--lon", type=float, default=77.2090)
    ap.add_argument("--radius", type=float, default=1500.0)
    ap.add_argument("--altitude", type=float, default=1200.0)
    ap.add_argument("--speed", type=float, default=45.0)
    ap.add_argument("--keep-video-only", action="store_true",
                    help="also keep the intermediate video-only TS")
    args = ap.parse_args()

    width, height = (int(x) for x in args.size.lower().split("x"))
    ff = ffmpeg.find("ffmpeg")
    print(f"ffmpeg    : {ff}")

    out = pathlib.Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".ts", dir=out.parent)
    import os
    os.close(tmp_fd)
    tmp = pathlib.Path(tmp_name)
    try:
        print(f"rendering : {width}x{height} @ {args.fps}fps, {args.duration:g}s, {args.bitrate}")
        render_video(ff, tmp, args.duration, args.fps, width, height,
                     args.bitrate, args.source, hud=not args.no_hud)

        path = OrbitPath(args.lat, args.lon, args.radius, args.altitude, args.speed)
        combined, info = remux_with_klv(tmp.read_bytes(), path.at,
                                        klv_rate=args.klv_rate)
        out.write_bytes(combined)

        if args.keep_video_only:
            (out.with_suffix(".video.ts")).write_bytes(tmp.read_bytes())
    finally:
        tmp.unlink(missing_ok=True)

    size = out.stat().st_size
    print(f"\nwrote {out}")
    print(f"  {size:,} bytes / {info['ts_packets']:,} TS packets / {info['duration']:.1f}s")
    print(f"  video PID 0x{info['video_pid']:04X} (stream_type 0x{info['video_stream_type']:02X}), "
          f"PCR on 0x{info['pcr_pid']:04X}")
    print(f"  KLV   PID 0x{info['klv_pid']:04X}, {info['klv_frames']} metadata frames "
          f"@ {args.klv_rate:g}/s")
    print(f"  avg bitrate {size * 8 / max(info['duration'], 1e-9) / 1e6:.2f} Mbps")
    print(f"\nstream it:  python stream_udp.py --file {out.relative_to(HERE).as_posix()}")
    print(f"play it:    ffplay udp://127.0.0.1:5601   (while streaming)")


if __name__ == "__main__":
    main()
