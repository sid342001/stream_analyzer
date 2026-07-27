"""Self-checks for the KLV encoder and the TS muxer/demuxer.

Runs standalone (python tests/test_roundtrip.py) or under pytest.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from uavstream.flight import OrbitPath                      # noqa: E402
from uavstream.klv import (UavTelemetry, checksum_ok, decode_to_dict,  # noqa: E402
                           ber_length, read_ber_length)
from uavstream.tsmux import TsMuxer, TS_PACKET_SIZE, _crc32_mpeg, encode_pcr  # noqa: E402
from uavstream.tsparse import (LiveDemux, iter_klv, iter_packets,  # noqa: E402
                               parse_pmt_pids, payload_offset, read_pcr)


def test_ber_roundtrip():
    for n in (0, 1, 127, 128, 255, 256, 65535, 1 << 20):
        assert read_ber_length(ber_length(n), 0)[0] == n


def test_klv_roundtrip():
    tel = UavTelemetry(latitude=28.6139, longitude=77.2090, altitude=1234.5,
                       heading=271.25, pitch=-7.5, roll=12.25,
                       frame_center_lat=-33.8688, frame_center_lon=151.2093)
    packet = tel.to_klv()
    assert checksum_ok(packet)
    values = decode_to_dict(packet)
    assert abs(values["Sensor Latitude"] - 28.6139) < 1e-5
    assert abs(values["Sensor Longitude"] - 77.2090) < 1e-5
    assert abs(values["Sensor True Altitude"] - 1234.5) < 0.5
    assert abs(values["Platform Heading Angle"] - 271.25) < 0.01
    assert abs(values["Platform Pitch Angle"] + 7.5) < 0.01
    assert abs(values["Frame Center Longitude"] - 151.2093) < 1e-5
    assert values["Platform Call Sign"] == "TOPGUN"


def test_klv_detects_corruption():
    packet = bytearray(UavTelemetry(latitude=10.0).to_klv())
    packet[40] ^= 0xFF
    assert not checksum_ok(bytes(packet))


def test_psi_crc_is_self_checking():
    # running the MPEG CRC over section+CRC must give zero
    mux = TsMuxer()
    for packet in (mux.pat(), mux.pmt()):
        start = payload_offset(packet) + 1          # skip the pointer_field
        section = packet[start:]
        length = ((section[1] & 0x0F) << 8) | section[2]
        assert _crc32_mpeg(section[:3 + length]) == 0


def test_pcr_roundtrip():
    for value in (0, 27_000_000, 12_345_678_901):
        pkt = b"\x47\x01\x01\x30" + bytes([7, 0x10]) + encode_pcr(value) + b"\xFF" * 176
        assert read_pcr(pkt[:TS_PACKET_SIZE]) == value


def test_ts_stream_roundtrip():
    mux = TsMuxer()
    path = OrbitPath(28.6139, 77.2090)
    blob = bytearray()
    sources = []
    for i in range(25):
        tel = path.at(i / 10)
        sources.append(tel)
        blob += mux.pat() + mux.pmt()
        for pkt in mux.klv_pes(tel.to_klv(), pts_90k=int(i * 9000), pcr_27mhz=int(i * 2_700_000)):
            blob += pkt

    data = bytes(blob)
    assert len(data) % TS_PACKET_SIZE == 0
    assert all(pkt[0] == 0x47 for pkt in iter_packets(data))
    assert parse_pmt_pids(data) == {0x0101: 0x06}

    decoded = list(iter_klv(data))
    assert len(decoded) == len(sources)
    for (pid, klv), tel in zip(decoded, sources):
        assert pid == 0x0101
        assert checksum_ok(klv)
        assert abs(decode_to_dict(klv)["Sensor Latitude"] - tel.latitude) < 1e-5


def test_live_demux_across_datagram_boundaries():
    mux = TsMuxer()
    path = OrbitPath(28.6139, 77.2090)
    blob = bytearray()
    for i in range(20):
        blob += mux.pat() + mux.pmt()
        for pkt in mux.klv_pes(path.at(i / 10).to_klv(), int(i * 9000), int(i * 2_700_000)):
            blob += pkt

    demux = LiveDemux()
    got = 0
    for i in range(0, len(blob), 1316):  # the datagram size used on the wire
        got += sum(1 for _ in demux.feed(bytes(blob[i:i + 1316])))
    assert got >= 19, got  # the final PES may still be open when input ends


def test_continuity_counters_increment():
    mux = TsMuxer()
    blob = bytearray()
    for i in range(10):
        blob += mux.pat() + mux.pmt()
        for pkt in mux.klv_pes(UavTelemetry().to_klv(), i * 9000, i * 2_700_000):
            blob += pkt

    seen: dict[int, int] = {}
    for pkt in iter_packets(bytes(blob)):
        pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
        cc = pkt[3] & 0x0F
        if pid in seen:
            assert cc == (seen[pid] + 1) & 0x0F, f"CC gap on pid 0x{pid:04X}"
        seen[pid] = cc


def test_pts_roundtrip():
    from uavstream.tsmux import encode_pts
    from uavstream.tsparse import read_pts
    for value in (0, 90_000, 3_000_000_000, (1 << 33) - 1):
        # a minimal PES on a packet: header + stream_id + len + flags + PTS
        optional = b"\x80\x80\x05" + encode_pts(value)
        pes = b"\x00\x00\x01\xBD" + bytes([0, len(optional)]) + optional
        pkt = bytearray(b"\x47\x40\x64\x10")  # pusi, pid 0x0064, payload only
        pkt += pes + b"\xFF" * (TS_PACKET_SIZE - len(pkt) - len(pes))
        assert read_pts(bytes(pkt[:TS_PACKET_SIZE])) == value


def test_remux_adds_klv_to_video():
    """Synthesise a tiny 'video' TS (a fake PES on a PCR PID) and remux KLV in."""
    from uavstream.remux import remux_with_klv
    from uavstream.tsparse import analyze, iter_klv, first_pts

    mux = TsMuxer(pmt_pid=0x1000, klv_pid=0x0100)  # klv_pid unused here
    video_pid = 0x0100
    blob = bytearray()
    for i in range(60):
        if i % 20 == 0:
            # PAT + a PMT advertising one H.264 video stream, PCR on the video PID
            blob += mux.pat()
            blob += mux.pmt_multi([(video_pid, 0x1B, b"")], pcr_pid=video_pid)
        # a video packet carrying PCR and (on the first) a PTS
        pcr = i * 900_000  # 30 ms steps in 27 MHz units
        payload = b"\x00\x00\x01\xE0\x00\x00\x80\x80\x05" + __import__(
            "uavstream.tsmux", fromlist=["encode_pts"]).encode_pts(i * 2700)
        blob += mux._packet(video_pid, payload, pusi=True, pcr_27mhz=pcr)

    path = OrbitPath(28.6139, 77.2090)
    combined, info = remux_with_klv(bytes(blob), path.at, klv_rate=10.0, klv_pid=0x0101)

    got = analyze(combined)
    assert got["streams"].get(0x0100) == 0x1B      # video preserved
    assert got["streams"].get(0x0101) == 0x06      # KLV added
    assert got["pcr_pid"] == 0x0100
    frames = list(iter_klv(combined))
    assert frames and all(pid == 0x0101 for pid, _ in frames)
    assert all(checksum_ok(k) for _, k in frames)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
