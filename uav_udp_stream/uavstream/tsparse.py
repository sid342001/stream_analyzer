"""Light-weight TS demux: find KLV PIDs, reassemble PES, read PCR."""

from __future__ import annotations

import struct
from collections.abc import Iterator

from .tsmux import TS_PACKET_SIZE, SYNC_BYTE, PID_PAT, STREAM_TYPE_KLV

# stream types that can legitimately carry KLV metadata
KLV_STREAM_TYPES = {STREAM_TYPE_KLV, 0x15}  # 0x06 private PES, 0x15 metadata in PES


def iter_packets(data: bytes) -> Iterator[bytes]:
    """Yield 188-byte packets, resynchronising on the 0x47 sync byte."""
    i = 0
    n = len(data)
    while i + TS_PACKET_SIZE <= n:
        if data[i] != SYNC_BYTE:
            i = data.find(SYNC_BYTE, i + 1)
            if i < 0:
                return
            continue
        yield data[i:i + TS_PACKET_SIZE]
        i += TS_PACKET_SIZE


def packet_pid(pkt: bytes) -> int:
    return ((pkt[1] & 0x1F) << 8) | pkt[2]


def payload_offset(pkt: bytes) -> int:
    """Start of the payload, or -1 when the packet carries none."""
    afc = (pkt[3] >> 4) & 0b11
    if afc in (0b00, 0b10):
        return -1
    if afc == 0b01:
        return 4
    return 5 + pkt[4]


def read_pts(pkt: bytes) -> int | None:
    """PTS (90 kHz units) from a packet that starts a PES with a PTS field."""
    if not pkt[1] & 0x40:  # needs payload_unit_start
        return None
    off = payload_offset(pkt)
    if off < 0:
        return None
    pes = pkt[off:]
    if len(pes) < 14 or pes[:3] != b"\x00\x00\x01" or not pes[7] & 0x80:
        return None
    p = pes[9:14]
    return (((p[0] >> 1) & 0x07) << 30 | p[1] << 22
            | ((p[2] >> 1) & 0x7F) << 15 | p[3] << 7 | (p[4] >> 1) & 0x7F)


def first_pts(data: bytes, pid: int) -> int | None:
    """First PTS seen on a PID, scanning from the start of the stream."""
    for pkt in iter_packets(data):
        if packet_pid(pkt) == pid:
            pts = read_pts(pkt)
            if pts is not None:
                return pts
    return None


def read_pcr(pkt: bytes) -> int | None:
    """PCR in 27 MHz units, if this packet carries one."""
    if (pkt[3] >> 4) & 0b10 == 0 or pkt[4] == 0 or not pkt[5] & 0x10:
        return None
    b = pkt[6:12]
    base = (b[0] << 25) | (b[1] << 17) | (b[2] << 9) | (b[3] << 1) | (b[4] >> 7)
    ext = ((b[4] & 0x01) << 8) | b[5]
    return base * 300 + ext


def parse_pmt_pids(data: bytes) -> dict[int, int]:
    """Scan a TS blob and return {elementary_pid: stream_type} for program 1..n."""
    pmt_pids: set[int] = set()
    streams: dict[int, int] = {}

    for pkt in iter_packets(data):
        pid = packet_pid(pkt)
        off = payload_offset(pkt)
        if off < 0 or off >= TS_PACKET_SIZE:
            continue
        if pkt[1] & 0x40:  # payload_unit_start: skip pointer_field
            off += 1 + pkt[off]
        section = pkt[off:]
        if not section:
            continue

        if pid == PID_PAT and section[0] == 0x00:
            length = ((section[1] & 0x0F) << 8) | section[2]
            body = section[8:3 + length - 4]
            for i in range(0, len(body) - 3, 4):
                prog = struct.unpack(">H", body[i:i + 2])[0]
                mapped = struct.unpack(">H", body[i + 2:i + 4])[0] & 0x1FFF
                if prog != 0:
                    pmt_pids.add(mapped)
        elif pid in pmt_pids and section[0] == 0x02:
            length = ((section[1] & 0x0F) << 8) | section[2]
            info_len = struct.unpack(">H", section[10:12])[0] & 0x0FFF
            i = 12 + info_len
            end = 3 + length - 4
            while i + 5 <= end <= len(section):
                stream_type = section[i]
                es_pid = struct.unpack(">H", section[i + 1:i + 3])[0] & 0x1FFF
                es_info_len = struct.unpack(">H", section[i + 3:i + 5])[0] & 0x0FFF
                streams[es_pid] = stream_type
                i += 5 + es_info_len
    return streams


def analyze(data: bytes) -> dict:
    """Return {pmt_pid, pcr_pid, streams:{pid: stream_type}} for program 1."""
    pmt_pids: set[int] = set()
    result: dict = {"pmt_pid": None, "pcr_pid": None, "streams": {}}

    for pkt in iter_packets(data):
        pid = packet_pid(pkt)
        off = payload_offset(pkt)
        if off < 0 or off >= TS_PACKET_SIZE:
            continue
        if pkt[1] & 0x40:
            off += 1 + pkt[off]
        section = pkt[off:]
        if not section:
            continue

        if pid == PID_PAT and section[0] == 0x00:
            length = ((section[1] & 0x0F) << 8) | section[2]
            body = section[8:3 + length - 4]
            for i in range(0, len(body) - 3, 4):
                prog = struct.unpack(">H", body[i:i + 2])[0]
                mapped = struct.unpack(">H", body[i + 2:i + 4])[0] & 0x1FFF
                if prog != 0:
                    pmt_pids.add(mapped)
        elif pid in pmt_pids and section[0] == 0x02 and result["pmt_pid"] is None:
            result["pmt_pid"] = pid
            length = ((section[1] & 0x0F) << 8) | section[2]
            result["pcr_pid"] = struct.unpack(">H", section[8:10])[0] & 0x1FFF
            info_len = struct.unpack(">H", section[10:12])[0] & 0x0FFF
            i = 12 + info_len
            end = 3 + length - 4
            while i + 5 <= end <= len(section):
                stream_type = section[i]
                es_pid = struct.unpack(">H", section[i + 1:i + 3])[0] & 0x1FFF
                es_info_len = struct.unpack(">H", section[i + 3:i + 5])[0] & 0x0FFF
                result["streams"][es_pid] = stream_type
                i += 5 + es_info_len

        if result["pmt_pid"] is not None and result["streams"]:
            break
    return result


class PesReassembler:
    """Collects TS payloads on one PID and yields complete PES payloads."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def push(self, pkt: bytes) -> Iterator[bytes]:
        off = payload_offset(pkt)
        if off < 0 or off >= TS_PACKET_SIZE:
            return
        if pkt[1] & 0x40:  # new PES starts here - flush the previous one
            if self._buf:
                out = self._extract(bytes(self._buf))
                if out:
                    yield out
            self._buf = bytearray(pkt[off:])
        elif self._buf:
            self._buf += pkt[off:]

        # a PES with a declared length can be emitted as soon as it is complete
        if len(self._buf) >= 6:
            declared = struct.unpack(">H", self._buf[4:6])[0]
            if declared and len(self._buf) >= declared + 6:
                out = self._extract(bytes(self._buf[:declared + 6]))
                self._buf = bytearray()
                if out:
                    yield out

    @staticmethod
    def _extract(pes: bytes) -> bytes | None:
        """Strip the PES header and return the elementary payload."""
        if len(pes) < 9 or pes[:3] != b"\x00\x00\x01":
            return None
        header_len = pes[8]
        return pes[9 + header_len:] or None


class LiveDemux:
    """Stateful demux for a continuous stream (e.g. arriving UDP datagrams)."""

    def __init__(self) -> None:
        self.streams: dict[int, int] = {}
        self.klv_pids: set[int] = set()
        self._asm: dict[int, PesReassembler] = {}
        self._tail = b""

    def feed(self, data: bytes) -> Iterator[tuple[int, bytes]]:
        """Feed raw bytes, yield (pid, klv_packet)."""
        buf = self._tail + data
        usable = len(buf) - (len(buf) % TS_PACKET_SIZE)
        self._tail = buf[usable:]
        buf = buf[:usable]

        found = parse_pmt_pids(buf)
        if found:
            self.streams.update(found)
            self.klv_pids = {pid for pid, st in self.streams.items()
                             if st in KLV_STREAM_TYPES}

        for pkt in iter_packets(buf):
            pid = packet_pid(pkt)
            if pid not in self.klv_pids:
                continue
            asm = self._asm.setdefault(pid, PesReassembler())
            for payload in asm.push(pkt):
                yield pid, payload


def iter_klv(data: bytes, klv_pids: set[int] | None = None) -> Iterator[tuple[int, bytes]]:
    """Yield (pid, klv_packet) for every KLV PES payload in a TS blob."""
    if klv_pids is None:
        klv_pids = {pid for pid, st in parse_pmt_pids(data).items()
                    if st in KLV_STREAM_TYPES}
    asm = {pid: PesReassembler() for pid in klv_pids}
    for pkt in iter_packets(data):
        pid = packet_pid(pkt)
        if pid in asm:
            for payload in asm[pid].push(pkt):
                yield pid, payload
