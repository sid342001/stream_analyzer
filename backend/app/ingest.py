"""Read the UAV UDP stream: decoded video frames + KLV telemetry.

Video decode uses PyAV (`av`) — the `uavstream` package (uav_udp_stream/) only
handles KLV, not H.264 pixel decoding; PyAV is the standard way to demux+decode
an MPEG-TS/UDP source into numpy frames. See uav_udp_stream/recieve_stream.py
for the exploratory script that first proved this approach works.

For KLV we deliberately do NOT use that script's `klvdata` dependency — this
project already has a tested, MISB-ST0601-tag-aware decoder in
uavstream.klv.decode_to_dict()/checksum_ok(), so we reuse that instead of
pulling in a second, generic KLV library. We also skip uavstream.tsparse's
`LiveDemux`/`PesReassembler`: those exist to reassemble PES packets from a raw
socket (see uav_udp_stream/recv_udp.py); PyAV already does that reassembly as
part of its own demuxing, so `packet.to_bytes()` on the "data" stream is
already a complete KLV Local Set ready for decode_to_dict().
"""

from __future__ import annotations

import logging

import av

from uavstream.klv import checksum_ok, decode_to_dict

logger = logging.getLogger(__name__)

# MISB ST 0601 tag names (see uavstream.klv.TAG_NAMES) -> our Telemetry schema
_TELEMETRY_KEYS = {
    "lat": "Sensor Latitude",
    "lon": "Sensor Longitude",
    "alt": "Sensor True Altitude",
    "hdg": "Platform Heading Angle",
    "speed": "Platform Ground Speed",
}


def _to_telemetry(values: dict[str, object]) -> dict[str, float] | None:
    """Map decoded KLV tag names onto the flat lat/lon/alt/hdg/speed shape.

    Returns None if any field is missing rather than guessing — a partial
    telemetry frame is worse than no update, since the UI trusts every field.
    """
    out: dict[str, float] = {}
    for field, tag_name in _TELEMETRY_KEYS.items():
        value = values.get(tag_name)
        if not isinstance(value, (int, float)):
            return None
        out[field] = float(value)
    return out


def frames(stream_url: str, open_timeout_s: float = 30.0, read_timeout_s: float = 5.0):
    """Yield (video_frame, telemetry) pairs from the live stream — exactly one
    of the pair is set per item: a PyAV `VideoFrame`, or a decoded telemetry
    dict.

    Yields the **undecoded PyAV frame**, not an ndarray, on purpose: converting
    to RGB is a full swscale pass plus an allocation, and at a 25-30fps source
    most frames are dropped by the display/detection rate gates. The caller
    decides whether a frame is wanted before paying for `to_ndarray()`.

    Blocking generator — run it in a background thread, not the asyncio event
    loop (see app/pipeline.py). The caller is responsible for remembering the
    last-seen telemetry between updates (this only reports what changed).

    `read_timeout_s` matters for shutdown: without it, a silent UDP source
    leaves `demux()` blocked in FFmpeg forever and the owning thread can never
    observe its stop event.

    `open_timeout_s` must stay well above FFmpeg's `analyzeduration` (below),
    because opening a live stream includes probing it for format. Passing a
    single scalar `timeout=` applies it to *both* phases, which makes a short
    read timeout also cap probing — that fails with
    `ExitError: Immediate exit requested` before the stream is ever opened.
    Hence the (open, read) tuple.
    """
    # fifo_size/overrun_nonfatal are udp:// protocol options, not generic
    # AVFormatContext options, so they belong on the URL query string, not in
    # `options=` below (same fix uav_udp_stream/recieve_stream.py landed on
    # independently) — without them a burst of datagrams can overrun the
    # kernel socket buffer and abort the whole demux with "Circular buffer
    # overrun", observed during testing.
    if stream_url.startswith("udp://") and "?" not in stream_url:
        # fifo_size: 5MB (up from 1MB). Now that the ingest thread never blocks
        # on inference this shouldn't overrun at all, but the headroom is free
        # and absorbs I-frame bursts.
        #
        # reuse=1 (SO_REUSEADDR) is not optional here, it's what makes retries
        # survivable. If av.open() raises *during* opening — e.g. the probe
        # times out because the source went briefly quiet — the exception comes
        # from av.open() itself, so the `with` block below is never entered and
        # its __exit__ never runs. The half-open container's socket is left
        # bound to this port with no Python reference able to close it, and
        # every later retry then fails to bind, permanently, until the process
        # restarts. SO_REUSEADDR lets the next attempt bind anyway.
        stream_url = f"{stream_url}?fifo_size=5000000&overrun_nonfatal=1&reuse=1"

    logger.info("opening stream: %s", stream_url)
    # `with` is required, not optional cleanup: if this generator is abandoned
    # mid-iteration (an exception anywhere downstream, the pipeline's retry
    # loop moving on, the caller breaking early), an unclosed av.Container
    # leaks its underlying UDP socket. The next attempt then fails to rebind
    # the same address ("Address already in use") until the process exits —
    # confirmed by hitting exactly that during testing.
    with av.open(
        stream_url,
        options={
            "fflags": "nobuffer",
            "flags": "low_delay",
            # Cut probing from FFmpeg's 5s default. We know what this stream
            # is (H.264 + KLV in MPEG-TS), so a long analyzeduration only
            # delays startup and every reconnect.
            "analyzeduration": "1000000",  # microseconds
            "probesize": "500000",         # bytes
        },
        timeout=(open_timeout_s, read_timeout_s),
    ) as container:
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        data_stream = next((s for s in container.streams if s.type == "data"), None)
        if video_stream is None:
            raise RuntimeError(f"no video stream found in {stream_url}")
        if data_stream is None:
            logger.warning("no KLV data stream found in %s — telemetry will stay empty", stream_url)

        for packet in container.demux():
            if packet.stream.type == "video":
                for frame in packet.decode():
                    yield frame, None
            elif packet.stream.type == "data":
                # av.Packet has no to_bytes() (that was recieve_stream.py's
                # assumption, unverified) — it implements the buffer protocol,
                # so bytes(packet) is the correct conversion. Confirmed against
                # the installed av==17.1.0.
                klv = bytes(packet)
                if not klv:
                    continue
                if not checksum_ok(klv):
                    logger.debug("KLV checksum failed, dropping packet")
                    continue
                telemetry = _to_telemetry(decode_to_dict(klv))
                if telemetry is not None:
                    yield None, telemetry
