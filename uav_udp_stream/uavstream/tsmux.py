"""Minimal MPEG-2 Transport Stream muxer for KLV metadata (MISB ST 1402 style).

Produces a conformant single-program TS carrying an asynchronous/synchronous
KLV metadata elementary stream:

    PID 0x0000  PAT
    PID 0x1000  PMT   (program 1)
    PID 0x0101  KLV   stream_type 0x06 (PES private data) + "KLVA" registration

PCR is carried on the KLV PID so the stream is self-timed even without video.
"""

from __future__ import annotations

import struct

TS_PACKET_SIZE = 188
SYNC_BYTE = 0x47

PID_PAT = 0x0000
PID_PMT = 0x1000
PID_KLV = 0x0101
PID_NULL = 0x1FFF

STREAM_TYPE_KLV = 0x06          # PES packets containing private data
PES_STREAM_ID_PRIVATE_1 = 0xBD  # what MISB / ffmpeg use for synchronous KLV

SYSTEM_CLOCK_HZ = 27_000_000
PTS_HZ = 90_000


def _crc32_mpeg(data: bytes) -> int:
    """MPEG-2 systems CRC-32 (poly 0x04C11DB7, MSB-first, init 0xFFFFFFFF)."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def encode_pts(pts_90k: int) -> bytes:
    """5-byte PTS field with '0010' prefix (PTS only, no DTS)."""
    pts_90k &= (1 << 33) - 1
    return bytes([
        0x21 | ((pts_90k >> 29) & 0x0E),
        (pts_90k >> 22) & 0xFF,
        0x01 | ((pts_90k >> 14) & 0xFE),
        (pts_90k >> 7) & 0xFF,
        0x01 | ((pts_90k << 1) & 0xFE),
    ])


def encode_pcr(pcr_27mhz: int) -> bytes:
    """6-byte PCR field: 33-bit base @90kHz, 6 reserved bits, 9-bit extension."""
    base = (pcr_27mhz // 300) & ((1 << 33) - 1)
    ext = pcr_27mhz % 300
    return bytes([
        (base >> 25) & 0xFF,
        (base >> 17) & 0xFF,
        (base >> 9) & 0xFF,
        (base >> 1) & 0xFF,
        ((base & 1) << 7) | 0x7E | ((ext >> 8) & 0x01),
        ext & 0xFF,
    ])


class TsMuxer:
    """Builds TS packets for a single-program KLV stream."""

    def __init__(self, pmt_pid: int = PID_PMT, klv_pid: int = PID_KLV,
                 program_number: int = 1) -> None:
        self.pmt_pid = pmt_pid
        self.klv_pid = klv_pid
        self.program_number = program_number
        self._cc: dict[int, int] = {}

    # -- low level ---------------------------------------------------------

    def _next_cc(self, pid: int) -> int:
        cc = self._cc.get(pid, 0)
        self._cc[pid] = (cc + 1) & 0x0F
        return cc

    def _packet(self, pid: int, payload: bytes, pusi: bool,
                pcr_27mhz: int | None = None) -> bytes:
        """Wrap up to one packet's worth of payload, stuffing via adaptation field."""
        max_payload = TS_PACKET_SIZE - 4 - (8 if pcr_27mhz is not None else 0)
        if len(payload) > max_payload:
            raise ValueError("payload does not fit in one TS packet")

        af_total = TS_PACKET_SIZE - 4 - len(payload)  # includes the length byte itself
        if af_total == 0:
            af, afc = b"", 0b01
        elif af_total == 1 and pcr_27mhz is None:
            af, afc = b"\x00", 0b11            # zero-length adaptation field
        else:
            flags = 0x10 if pcr_27mhz is not None else 0x00
            body = encode_pcr(pcr_27mhz) if pcr_27mhz is not None else b""
            body += b"\xFF" * (af_total - 2 - len(body))
            af, afc = bytes([af_total - 1, flags]) + body, 0b11

        header = struct.pack(
            ">BHB",
            SYNC_BYTE,
            (0x4000 if pusi else 0x0000) | pid,
            (afc << 4) | self._next_cc(pid),
        )
        packet = header + af + payload
        assert len(packet) == TS_PACKET_SIZE, len(packet)
        return packet

    def _section_packet(self, pid: int, section: bytes) -> bytes:
        """PSI sections here always fit in one packet (pointer_field = 0)."""
        return self._packet(pid, b"\x00" + section, pusi=True)

    # -- PSI ---------------------------------------------------------------

    def pat(self) -> bytes:
        body = struct.pack(">HH", self.program_number, 0xE000 | self.pmt_pid)
        section = self._psi_section(table_id=0x00, table_id_ext=0x0001, body=body)
        return self._section_packet(PID_PAT, section)

    def pmt(self) -> bytes:
        # registration_descriptor: format_identifier "KLVA"
        descriptors = b"\x05\x04KLVA"
        es_info = struct.pack(">BHH", STREAM_TYPE_KLV,
                              0xE000 | self.klv_pid,
                              0xF000 | len(descriptors)) + descriptors
        body = struct.pack(">HH", 0xE000 | self.klv_pid, 0xF000) + es_info
        section = self._psi_section(table_id=0x02, table_id_ext=self.program_number, body=body)
        return self._section_packet(self.pmt_pid, section)

    def pmt_multi(self, streams: list[tuple[int, int, bytes]], pcr_pid: int) -> bytes:
        """PMT for an arbitrary set of (es_pid, stream_type, descriptors) streams."""
        body = struct.pack(">HH", 0xE000 | pcr_pid, 0xF000)  # PCR_PID, prog_info_len=0
        for es_pid, stream_type, desc in streams:
            body += struct.pack(">BHH", stream_type, 0xE000 | es_pid,
                                0xF000 | len(desc)) + desc
        section = self._psi_section(table_id=0x02, table_id_ext=self.program_number, body=body)
        return self._section_packet(self.pmt_pid, section)

    @staticmethod
    def _psi_section(table_id: int, table_id_ext: int, body: bytes,
                     version: int = 0) -> bytes:
        # 5 bytes of syntax header + body + 4 byte CRC
        section_length = 5 + len(body) + 4
        header = struct.pack(
            ">BHHBBB",
            table_id,
            0xB000 | section_length,  # section_syntax_indicator=1, '0', reserved
            table_id_ext,
            0xC1 | (version << 1),    # reserved, version, current_next_indicator=1
            0x00,                     # section_number
            0x00,                     # last_section_number
        )
        section = header + body
        return section + struct.pack(">I", _crc32_mpeg(section))

    # -- PES ---------------------------------------------------------------

    def klv_pes(self, klv: bytes, pts_90k: int, pcr_27mhz: int | None = None) -> list[bytes]:
        """Packetise one KLV Local Set into TS packets on the KLV PID."""
        optional = b"\x84\x80\x05" + encode_pts(pts_90k)  # data_alignment=1, PTS only
        pes = (b"\x00\x00\x01" + bytes([PES_STREAM_ID_PRIVATE_1])
               + struct.pack(">H", len(optional) + len(klv)) + optional + klv)

        packets: list[bytes] = []
        offset = 0
        first = True
        while offset < len(pes):
            pcr = pcr_27mhz if first else None
            room = TS_PACKET_SIZE - 4 - (8 if pcr is not None else 0)
            chunk = pes[offset:offset + room]
            packets.append(self._packet(self.klv_pid, chunk, pusi=first, pcr_27mhz=pcr))
            offset += len(chunk)
            first = False
        return packets

    @staticmethod
    def null_packet() -> bytes:
        return bytes([SYNC_BYTE, 0x1F, 0xFF, 0x10]) + b"\xFF" * 184
