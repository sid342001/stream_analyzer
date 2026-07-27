"""MISB ST 0601 UAS Datalink Local Set - encoder / decoder.

Pure stdlib. Values are stored in the Local Set as scaled integers; this module
converts to and from natural units (degrees, metres, m/s).
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field

# SMPTE 336M Universal Key for the UAS Datalink Local Set (ST 0601).
UAS_LDS_KEY = bytes(
    [0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01,
     0x0E, 0x01, 0x03, 0x01, 0x01, 0x00, 0x00, 0x00]
)

TAG_CHECKSUM = 1
TAG_TIMESTAMP = 2
TAG_MISSION_ID = 3
TAG_PLATFORM_TAIL = 4
TAG_PLATFORM_HEADING = 5
TAG_PLATFORM_PITCH = 6
TAG_PLATFORM_ROLL = 7
TAG_PLATFORM_DESIGNATION = 10
TAG_IMAGE_SOURCE_SENSOR = 11
TAG_IMAGE_COORD_SYSTEM = 12
TAG_SENSOR_LATITUDE = 13
TAG_SENSOR_LONGITUDE = 14
TAG_SENSOR_TRUE_ALTITUDE = 15
TAG_SENSOR_HFOV = 16
TAG_SENSOR_VFOV = 17
TAG_SENSOR_REL_AZIMUTH = 18
TAG_SENSOR_REL_ELEVATION = 19
TAG_SENSOR_REL_ROLL = 20
TAG_SLANT_RANGE = 21
TAG_TARGET_WIDTH = 22
TAG_FRAME_CENTER_LAT = 23
TAG_FRAME_CENTER_LON = 24
TAG_FRAME_CENTER_ELEV = 25
TAG_TARGET_LOCATION_LAT = 40
TAG_TARGET_LOCATION_LON = 41
TAG_TARGET_LOCATION_ELEV = 42
TAG_PLATFORM_GROUND_SPEED = 56
TAG_PLATFORM_CALL_SIGN = 59
TAG_LDS_VERSION = 65

TAG_NAMES = {
    1: "Checksum", 2: "UNIX Time Stamp", 3: "Mission ID", 4: "Platform Tail Number",
    5: "Platform Heading Angle", 6: "Platform Pitch Angle", 7: "Platform Roll Angle",
    10: "Platform Designation", 11: "Image Source Sensor", 12: "Image Coordinate System",
    13: "Sensor Latitude", 14: "Sensor Longitude", 15: "Sensor True Altitude",
    16: "Sensor Horizontal FOV", 17: "Sensor Vertical FOV", 18: "Sensor Rel Azimuth Angle",
    19: "Sensor Rel Elevation Angle", 20: "Sensor Rel Roll Angle", 21: "Slant Range",
    22: "Target Width", 23: "Frame Center Latitude", 24: "Frame Center Longitude",
    25: "Frame Center Elevation", 40: "Target Location Latitude",
    41: "Target Location Longitude", 42: "Target Location Elevation",
    56: "Platform Ground Speed", 59: "Platform Call Sign", 65: "UAS LDS Version Number",
}

# ---------------------------------------------------------------------------
# scaling helpers
# ---------------------------------------------------------------------------

_INT32_MAX = 2 ** 31 - 1
_UINT16_MAX = 2 ** 16 - 1
_UINT32_MAX = 2 ** 32 - 1
_INT16_MAX = 2 ** 15 - 1


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _enc_int(value: float, lo: float, hi: float, nbytes: int, signed: bool) -> bytes:
    """Map a float in [lo, hi] onto the full range of an n-byte integer."""
    value = _clamp(value, lo, hi)
    if signed:
        full = 2 ** (8 * nbytes - 1) - 1
        raw = round(value * full / max(abs(lo), abs(hi)))
    else:
        full = 2 ** (8 * nbytes) - 1
        raw = round((value - lo) * full / (hi - lo))
    return raw.to_bytes(nbytes, "big", signed=signed)


def _dec_int(raw: bytes, lo: float, hi: float, signed: bool) -> float:
    n = int.from_bytes(raw, "big", signed=signed)
    if signed:
        full = 2 ** (8 * len(raw) - 1) - 1
        return n * max(abs(lo), abs(hi)) / full
    full = 2 ** (8 * len(raw)) - 1
    return lo + n * (hi - lo) / full


def enc_lat(deg: float) -> bytes:
    return _enc_int(deg, -90.0, 90.0, 4, True)


def enc_lon(deg: float) -> bytes:
    return _enc_int(deg, -180.0, 180.0, 4, True)


def enc_alt(metres: float) -> bytes:
    return _enc_int(metres, -900.0, 19000.0, 2, False)


def enc_angle360_u16(deg: float) -> bytes:
    return _enc_int(deg % 360.0, 0.0, 360.0, 2, False)


def enc_angle360_u32(deg: float) -> bytes:
    return _enc_int(deg % 360.0, 0.0, 360.0, 4, False)


# ---------------------------------------------------------------------------
# BER / checksum
# ---------------------------------------------------------------------------

def ber_length(n: int) -> bytes:
    """BER short/long form length encoding."""
    if n < 0x80:
        return bytes([n])
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


def read_ber_length(buf: bytes, off: int) -> tuple[int, int]:
    """Return (length, new_offset)."""
    first = buf[off]
    off += 1
    if first < 0x80:
        return first, off
    n = first & 0x7F
    return int.from_bytes(buf[off:off + n], "big"), off + n


def bcc_16(data: bytes) -> int:
    """ST 0601 checksum: running 16-bit sum of the packet, big-endian aligned."""
    total = 0
    for i, byte in enumerate(data):
        total += byte << (8 * ((i + 1) % 2))
    return total & 0xFFFF


# ---------------------------------------------------------------------------
# telemetry model
# ---------------------------------------------------------------------------

@dataclass
class UavTelemetry:
    """One frame of UAV telemetry, in natural units."""

    timestamp_us: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    mission_id: str = "MISSION01"
    platform_tail: str = "UAV-01"
    platform_designation: str = "Predator"
    image_source_sensor: str = "EO Nose"
    image_coord_system: str = "Geodetic WGS84"
    call_sign: str = "TOPGUN"

    # platform
    latitude: float = 0.0            # deg
    longitude: float = 0.0           # deg
    altitude: float = 1000.0         # m MSL
    heading: float = 0.0             # deg
    pitch: float = 0.0               # deg  (-20..20)
    roll: float = 0.0                # deg  (-50..50)
    ground_speed: float = 45.0       # m/s  (0..255)

    # sensor / gimbal
    hfov: float = 12.0               # deg
    vfov: float = 7.0                # deg
    rel_azimuth: float = 0.0         # deg
    rel_elevation: float = -30.0     # deg
    rel_roll: float = 0.0            # deg
    slant_range: float = 2000.0      # m

    # what the sensor is looking at
    frame_center_lat: float = 0.0
    frame_center_lon: float = 0.0
    frame_center_elev: float = 0.0
    target_width: float = 250.0      # m

    def to_klv(self) -> bytes:
        """Serialise as a complete ST 0601 Local Set packet (key + len + TLVs)."""
        items: list[tuple[int, bytes]] = [
            (TAG_TIMESTAMP, struct.pack(">Q", self.timestamp_us)),
            (TAG_MISSION_ID, self.mission_id.encode("ascii")[:127]),
            (TAG_PLATFORM_TAIL, self.platform_tail.encode("ascii")[:127]),
            (TAG_PLATFORM_HEADING, enc_angle360_u16(self.heading)),
            (TAG_PLATFORM_PITCH, _enc_int(self.pitch, -20.0, 20.0, 2, True)),
            (TAG_PLATFORM_ROLL, _enc_int(self.roll, -50.0, 50.0, 2, True)),
            (TAG_PLATFORM_DESIGNATION, self.platform_designation.encode("ascii")[:127]),
            (TAG_IMAGE_SOURCE_SENSOR, self.image_source_sensor.encode("ascii")[:127]),
            (TAG_IMAGE_COORD_SYSTEM, self.image_coord_system.encode("ascii")[:127]),
            (TAG_SENSOR_LATITUDE, enc_lat(self.latitude)),
            (TAG_SENSOR_LONGITUDE, enc_lon(self.longitude)),
            (TAG_SENSOR_TRUE_ALTITUDE, enc_alt(self.altitude)),
            (TAG_SENSOR_HFOV, _enc_int(self.hfov, 0.0, 180.0, 2, False)),
            (TAG_SENSOR_VFOV, _enc_int(self.vfov, 0.0, 180.0, 2, False)),
            (TAG_SENSOR_REL_AZIMUTH, enc_angle360_u32(self.rel_azimuth)),
            (TAG_SENSOR_REL_ELEVATION, _enc_int(self.rel_elevation, -180.0, 180.0, 4, True)),
            (TAG_SENSOR_REL_ROLL, enc_angle360_u32(self.rel_roll)),
            (TAG_SLANT_RANGE, _enc_int(self.slant_range, 0.0, 5_000_000.0, 4, False)),
            (TAG_TARGET_WIDTH, _enc_int(self.target_width, 0.0, 10_000.0, 2, False)),
            (TAG_FRAME_CENTER_LAT, enc_lat(self.frame_center_lat)),
            (TAG_FRAME_CENTER_LON, enc_lon(self.frame_center_lon)),
            (TAG_FRAME_CENTER_ELEV, enc_alt(self.frame_center_elev)),
            (TAG_PLATFORM_GROUND_SPEED, bytes([int(_clamp(self.ground_speed, 0, 255))])),
            (TAG_PLATFORM_CALL_SIGN, self.call_sign.encode("ascii")[:127]),
            (TAG_LDS_VERSION, bytes([16])),
        ]

        body = b"".join(bytes([tag]) + ber_length(len(val)) + val for tag, val in items)
        # + 4 bytes for the checksum TLV that closes every ST 0601 packet
        packet = UAS_LDS_KEY + ber_length(len(body) + 4) + body + b"\x01\x02"
        return packet + struct.pack(">H", bcc_16(packet))


def decode_klv(data: bytes) -> dict[int, bytes]:
    """Decode one ST 0601 Local Set packet into {tag: raw_value}. Empty if invalid."""
    if not data.startswith(UAS_LDS_KEY):
        return {}
    length, off = read_ber_length(data, len(UAS_LDS_KEY))
    end = off + length
    if end > len(data):
        return {}
    out: dict[int, bytes] = {}
    while off < end:
        tag = data[off]
        vlen, off = read_ber_length(data, off + 1)
        out[tag] = data[off:off + vlen]
        off += vlen
    return out


def checksum_ok(data: bytes) -> bool:
    """Verify the trailing ST 0601 checksum of a Local Set packet."""
    if len(data) < 4:
        return False
    stated = struct.unpack(">H", data[-2:])[0]
    return bcc_16(data[:-2]) == stated


def decode_to_dict(data: bytes) -> dict[str, object]:
    """Decode a packet into human-readable values keyed by tag name."""
    raw = decode_klv(data)
    out: dict[str, object] = {}
    for tag, val in sorted(raw.items()):
        name = TAG_NAMES.get(tag, f"Tag {tag}")
        try:
            out[name] = _humanise(tag, val)
        except Exception:
            out[name] = val.hex()
    return out


def _humanise(tag: int, val: bytes):
    if tag == TAG_TIMESTAMP:
        return struct.unpack(">Q", val)[0]
    if tag in (TAG_SENSOR_LATITUDE, TAG_FRAME_CENTER_LAT, TAG_TARGET_LOCATION_LAT):
        return round(_dec_int(val, -90.0, 90.0, True), 7)
    if tag in (TAG_SENSOR_LONGITUDE, TAG_FRAME_CENTER_LON, TAG_TARGET_LOCATION_LON):
        return round(_dec_int(val, -180.0, 180.0, True), 7)
    if tag in (TAG_SENSOR_TRUE_ALTITUDE, TAG_FRAME_CENTER_ELEV, TAG_TARGET_LOCATION_ELEV):
        return round(_dec_int(val, -900.0, 19000.0, False), 2)
    if tag == TAG_PLATFORM_HEADING:
        return round(_dec_int(val, 0.0, 360.0, False), 3)
    if tag == TAG_PLATFORM_PITCH:
        return round(_dec_int(val, -20.0, 20.0, True), 3)
    if tag == TAG_PLATFORM_ROLL:
        return round(_dec_int(val, -50.0, 50.0, True), 3)
    if tag in (TAG_SENSOR_HFOV, TAG_SENSOR_VFOV):
        return round(_dec_int(val, 0.0, 180.0, False), 3)
    if tag in (TAG_SENSOR_REL_AZIMUTH, TAG_SENSOR_REL_ROLL):
        return round(_dec_int(val, 0.0, 360.0, False), 3)
    if tag == TAG_SENSOR_REL_ELEVATION:
        return round(_dec_int(val, -180.0, 180.0, True), 3)
    if tag == TAG_SLANT_RANGE:
        return round(_dec_int(val, 0.0, 5_000_000.0, False), 2)
    if tag == TAG_TARGET_WIDTH:
        return round(_dec_int(val, 0.0, 10_000.0, False), 2)
    if tag in (TAG_PLATFORM_GROUND_SPEED, TAG_LDS_VERSION):
        return val[0]
    if tag == TAG_CHECKSUM:
        return f"0x{val.hex()}"
    if tag in (TAG_MISSION_ID, TAG_PLATFORM_TAIL, TAG_PLATFORM_DESIGNATION,
               TAG_IMAGE_SOURCE_SENSOR, TAG_IMAGE_COORD_SYSTEM, TAG_PLATFORM_CALL_SIGN):
        return val.decode("ascii", errors="replace")
    return val.hex()
