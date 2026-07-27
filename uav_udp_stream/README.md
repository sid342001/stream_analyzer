# uav_udp_stream

UDP MPEG-TS streaming of a UAV feed — **H.264 video + MISB ST 0601 KLV
telemetry** (lat/long and full sensor metadata) in one synchronised transport
stream. Stream it over UDP to your real NIC IP **and** a pseudo IP at once, play
the video in VLC/ffplay, and decode the telemetry on the receiving side.

The TS muxer, KLV codec and UDP tooling are pure standard library. Video
generation uses **ffmpeg** (installed via `winget install Gyan.FFmpeg`); a
metadata-only mode still works with no ffmpeg at all.

```
uav_udp_stream/
├── .venv/                  virtual environment
├── config.json             stream targets (real IP + pseudo IP) and defaults
├── make_video_stream.py    build a .ts with H.264 video + KLV (needs ffmpeg)
├── make_stream.py          build a metadata-only .ts (no ffmpeg needed)
├── stream_udp.py           send a .ts over UDP to N targets, PCR-paced
├── recv_udp.py             receive UDP, demux KLV, print/log telemetry
├── add_pseudo_ip.ps1       add a secondary "pseudo" IP to a NIC (needs admin)
├── requirements.txt        optional extras only
├── streams/                generated + captured streams
├── tests/test_roundtrip.py self-checks
└── uavstream/
    ├── klv.py              ST 0601 Local Set encoder/decoder + checksum
    ├── tsmux.py            MPEG-TS muxer (PAT/PMT/PES/PCR)
    ├── tsparse.py          TS demux, PES reassembly, PCR/PTS readers
    ├── remux.py            fold KLV into an existing video TS, PTS-aligned
    ├── ffmpeg.py           locate the ffmpeg / ffprobe binaries
    └── flight.py           synthetic orbit flight path
```

## Setup

The venv already exists. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

The Python side needs no pip packages. ffmpeg 8.x is already installed via
winget; if a fresh shell can't find it, the tools also look in the winget
install folder automatically, or set `$env:FFMPEG` to `ffmpeg.exe`.

## Quick start (video + telemetry)

```powershell
# 1. build a 30 s feed: H.264 720p30 video + 10 KLV packets/sec, orbiting 28.6139N 77.2090E
python make_video_stream.py --duration 30 --fps 30 --klv-rate 10

# 2. in one terminal, listen on the pseudo IP and log telemetry
python recv_udp.py --port 5601 --json streams/telemetry.jsonl

# 3. in another, stream to both targets from config.json
python stream_udp.py

# 4. (optional) watch the actual video while it streams
ffplay udp://127.0.0.1:5601
```

Metadata-only, no ffmpeg required:

```powershell
python make_stream.py --duration 60 --rate 10 --out streams/uav_demo.ts
python stream_udp.py --file streams/uav_demo.ts
```

Receiver output:

```
#1     0x0101  lat=28.627375  lon=77.209000  alt=1200m  hdg=90.0deg  fc_lat=28.613900  fc_lon=77.209409  range=1803m
#2     0x0101  lat=28.627375  lon=77.209046  alt=1200m  hdg=90.2deg  fc_lat=28.613925  fc_lon=77.209409  range=1803m
```

## Streaming to your IP and a pseudo IP

`config.json` holds the destinations; every datagram is sent to all of them:

```json
"targets": [
  { "name": "real-nic",        "host": "192.168.1.127", "port": 5600 },
  { "name": "pseudo-loopback", "host": "127.0.0.1",     "port": 5601 }
]
```

Override from the command line at any time:

```powershell
python stream_udp.py --target 192.168.1.127:5600 --target 127.0.0.1:5601
python stream_udp.py --target 239.1.1.1:5600 --ttl 4 --iface 192.168.1.127   # multicast
```

Three ways to get a "pseudo" IP, in increasing order of realism:

1. **Loopback** — `127.0.0.1` on a different port. Zero setup, already in
   `config.json`.
2. **NIC alias** — a second real address on your existing adapter. Run an
   elevated PowerShell:
   ```powershell
   .\add_pseudo_ip.ps1 -IPAddress 10.10.10.10 -InterfaceAlias 'Ethernet'
   python stream_udp.py --target 10.10.10.10:5600
   python recv_udp.py --bind 10.10.10.10 --port 5600
   .\add_pseudo_ip.ps1 -IPAddress 10.10.10.10 -Remove     # clean up
   ```
3. **Multicast** — `239.x.x.x`, which any number of receivers can subscribe to:
   ```powershell
   python stream_udp.py --target 239.1.1.1:5600 --iface 192.168.1.127
   python recv_udp.py --group 239.1.1.1 --iface 192.168.1.127 --port 5600
   ```

## Stream format

Combined video feed from `make_video_stream.py`:

| PID | Contents |
|-----|----------|
| `0x0000` | PAT, program 1 |
| `0x1000` | PMT |
| `0x0100` | H.264 video, `stream_type 0x1B`, carries the PCR |
| `0x0101` | KLV, `stream_type 0x06` + `KLVA` registration descriptor, PES `stream_id 0xBD` |

Metadata-only feed from `make_stream.py` is the same minus the video PID, with
the PCR on the KLV PID. Either way it's the MISB ST 1402 layout, so VLC,
ffprobe, tsduck and typical UAV/FMV analysers recognise both tracks (verified
here: `ffprobe` reports `h264` + `klv (KLVA)`). Each KLV PES payload is one
complete ST 0601 Local Set: 16-byte UAS Datalink universal key, BER length,
tags, and a trailing 16-bit checksum. KLV PTS is aligned to the video
presentation clock so metadata tracks the correct frame.

Metadata carried per frame: UTC timestamp, mission ID, tail number, platform
designation, call sign, sensor lat/lon/altitude, platform heading/pitch/roll,
ground speed, sensor H/V FOV, relative azimuth/elevation/roll, slant range,
target width, frame centre lat/lon/elevation, and LDS version.

## Command reference

**make_video_stream.py** (needs ffmpeg)

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `streams/uav_video.ts` | output file |
| `--source` | — | use a real video file instead of the synthetic feed |
| `--duration` | `30` | seconds |
| `--fps` | `30` | video frame rate |
| `--size` | `1280x720` | resolution |
| `--bitrate` | `4M` | video bitrate |
| `--klv-rate` | `10` | KLV packets per second |
| `--no-hud` | off | skip the burnt-in crosshair/HUD text |
| `--lat` `--lon` `--radius` `--altitude` `--speed` | orbit params, as below |
| `--keep-video-only` | off | also keep the intermediate video-only TS |

To wrap **your own** UAV video with telemetry: `--source path\to\real.mp4`.

**make_stream.py** (metadata only, no ffmpeg)

| Flag | Default | Meaning |
|---|---|---|
| `--out` | `streams/uav_demo.ts` | output file |
| `--duration` | `60` | seconds |
| `--rate` | `10` | KLV packets per second |
| `--lat` `--lon` | `28.6139` `77.2090` | orbit centre |
| `--radius` `--altitude` `--speed` | `1500` `1200` `45` | m, m MSL, m/s |
| `--target-elev` | `200` | ground target elevation (m) |
| `--cbr` | `0` | pad with null packets to a constant bitrate (bps) |

**stream_udp.py**

| Flag | Default | Meaning |
|---|---|---|
| `--file` | from config | `.ts` to send (your own UAV captures work too) |
| `--target` | from config | `host:port`, repeatable |
| `--bitrate` | `2000000` | fallback pacing when the file has no PCR |
| `--packets` | `7` | TS packets per datagram (7 × 188 = 1316 bytes) |
| `--ttl` `--iface` | `8` / auto | multicast TTL and source interface |
| `--loop` | off | repeat forever |
| `--asap` | off | ignore timing, send at full speed |

**recv_udp.py**

| Flag | Default | Meaning |
|---|---|---|
| `--bind` `--port` | `0.0.0.0` `5600` | listen address |
| `--group` `--iface` | — | multicast group to join |
| `--save` | — | write the received TS to a file |
| `--json` | — | append decoded telemetry as JSON lines |
| `--full` | off | print every KLV tag instead of a one-line summary |
| `--count` `--timeout` | `0` `0` | stop after N packets / N idle seconds |

## Notes

- Timing follows the PCR in the file, so the stream leaves at true real-time
  rate. Windows' 15 ms sleep granularity is worked around with a short spin.
- 1316-byte datagrams keep you inside a 1500-byte MTU — no IP fragmentation.
- `stream_udp.py` will happily send any existing `.ts`, not just generated ones,
  and `recv_udp.py` will decode KLV from any stream that declares it in the PMT.
- `remux.py` keeps the original video packets and their PCR untouched and only
  rewrites the PAT/PMT to add the KLV track, so any ffmpeg-produced TS (or a real
  UAV capture via `--source`) can be enriched with telemetry without re-encoding
  the video.

## Tests

```powershell
python tests/test_roundtrip.py     # or: pytest tests/
```
