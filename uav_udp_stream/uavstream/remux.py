"""Fold a KLV metadata stream into an existing video transport stream.

Takes an MPEG-TS produced by ffmpeg (H.264 video with PCR) and interleaves a
MISB ST 0601 KLV elementary stream on a new PID, with each KLV packet's PTS
aligned to the video presentation clock. The original video packets and their
timing are preserved untouched; only the PAT/PMT are rewritten to advertise the
extra metadata track.
"""

from __future__ import annotations

from collections.abc import Callable

from .klv import UavTelemetry
from .tsmux import TsMuxer, TS_PACKET_SIZE, STREAM_TYPE_KLV
from .tsparse import analyze, iter_packets, packet_pid, read_pcr

PCR_WRAP = (1 << 33) * 300
# elementary stream types that carry video essence
VIDEO_STREAM_TYPES = {0x01, 0x02, 0x1B, 0x24}  # MPEG-1/2, H.264, HEVC


def _packet_timeline(packets: list[bytes], pcr_pid: int) -> list[float]:
    """Seconds-from-first-PCR for every packet, interpolated between PCRs."""
    marks: list[tuple[int, int]] = []
    last_raw = None
    wraps = 0
    for idx, pkt in enumerate(packets):
        if packet_pid(pkt) != pcr_pid:
            continue
        pcr = read_pcr(pkt)
        if pcr is None:
            continue
        if last_raw is not None and pcr < last_raw - PCR_WRAP // 2:
            wraps += 1
        last_raw = pcr
        marks.append((idx, pcr + wraps * PCR_WRAP))

    n = len(packets)
    if len(marks) < 2:  # no usable PCR - fall back to a flat 0..0 timeline
        return [0.0] * n

    base = marks[0][1]
    times = [0.0] * n
    for (i0, p0), (i1, p1) in zip(marks, marks[1:]):
        t0, t1 = (p0 - base) / 27e6, (p1 - base) / 27e6
        span = max(i1 - i0, 1)
        for k in range(i0, i1):
            times[k] = t0 + (t1 - t0) * (k - i0) / span
    rate = (times[marks[1][0]] - times[marks[0][0]]) / max(marks[1][0] - marks[0][0], 1)
    for k in range(0, marks[0][0]):
        times[k] = times[marks[0][0]] - rate * (marks[0][0] - k)
    last_idx = marks[-1][0]
    times[last_idx] = (marks[-1][1] - base) / 27e6
    for k in range(last_idx + 1, n):
        times[k] = times[last_idx] + rate * (k - last_idx)
    return times


def remux_with_klv(video_ts: bytes,
                   telemetry_at: Callable[[float], UavTelemetry],
                   klv_rate: float = 10.0,
                   klv_pid: int = 0x0101) -> tuple[bytes, dict]:
    """Return (combined_ts, info). `telemetry_at(t)` yields telemetry for time t."""
    info = analyze(video_ts)
    pmt_pid = info["pmt_pid"] or 0x1000
    pcr_pid = info["pcr_pid"] or 0x0100

    video_streams = [(pid, st) for pid, st in info["streams"].items()
                     if st in VIDEO_STREAM_TYPES]
    if not video_streams:
        raise ValueError("no video elementary stream found in the input TS")
    if klv_pid in info["streams"]:
        klv_pid = max(info["streams"]) + 1

    packets = list(iter_packets(video_ts))
    times = _packet_timeline(packets, pcr_pid)
    duration = times[-1] if times else 0.0

    from .tsparse import first_pts
    pts_epoch = first_pts(video_ts, video_streams[0][0]) or 0

    # advertise every original stream plus the new KLV track
    pmt_streams: list[tuple[int, int, bytes]] = []
    for pid, st in sorted(info["streams"].items()):
        pmt_streams.append((pid, st, b""))
    pmt_streams.append((klv_pid, STREAM_TYPE_KLV, b"\x05\x04KLVA"))

    mux = TsMuxer(pmt_pid=pmt_pid, klv_pid=klv_pid)

    frames = int(duration * klv_rate) if duration > 0 else 0
    klv_times = [i / klv_rate for i in range(frames)]

    out = bytearray()
    ki = 0
    for idx, pkt in enumerate(packets):
        pid = packet_pid(pkt)
        if pid == 0x0000:
            out += mux.pat()
        elif pid == pmt_pid:
            out += mux.pmt_multi(pmt_streams, pcr_pid)
        else:
            out += pkt

        while ki < frames and klv_times[ki] <= times[idx]:
            t = klv_times[ki]
            pts = int(pts_epoch + t * 90_000)
            for kp in mux.klv_pes(telemetry_at(t).to_klv(), pts_90k=pts):
                out += kp
            ki += 1

    # any KLV scheduled past the last video packet
    while ki < frames:
        t = klv_times[ki]
        pts = int(pts_epoch + t * 90_000)
        for kp in mux.klv_pes(telemetry_at(t).to_klv(), pts_90k=pts):
            out += kp
        ki += 1

    summary = {
        "video_pid": video_streams[0][0],
        "video_stream_type": video_streams[0][1],
        "pmt_pid": pmt_pid,
        "pcr_pid": pcr_pid,
        "klv_pid": klv_pid,
        "duration": duration,
        "klv_frames": frames,
        "ts_packets": len(out) // TS_PACKET_SIZE,
    }
    return bytes(out), summary
