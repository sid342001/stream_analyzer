# # #!/usr/bin/env python3

# # import argparse
# # import socket
# # import signal
# # import sys
# # import time


# # def main():
# #     parser = argparse.ArgumentParser(description="UDP MPEG-TS Receiver")
# #     parser.add_argument("--host", default="0.0.0.0",
# #                         help="Host/IP to bind (default: 0.0.0.0)")
# #     parser.add_argument("--port", type=int, default=5601,
# #                         help="UDP port (default: 5601)")
# #     parser.add_argument("--output", help="Save received TS to file")
# #     parser.add_argument("--buffer", type=int, default=65535,
# #                         help="Socket receive buffer size")
# #     args = parser.parse_args()

# #     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# #     sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.buffer)
# #     sock.bind((args.host, args.port))

# #     outfile = None
# #     if args.output:
# #         outfile = open(args.output, "wb")

# #     running = True

# #     def stop(sig, frame):
# #         nonlocal running
# #         running = False

# #     signal.signal(signal.SIGINT, stop)

# #     print(f"Listening on udp://{args.host}:{args.port}")
# #     print("Press Ctrl+C to stop\n")

# #     datagrams = 0
# #     bytes_received = 0
# #     bad_packets = 0

# #     start = time.time()
# #     last = start

# #     while running:
# #         try:
# #             data, addr = sock.recvfrom(65535)
# #         except KeyboardInterrupt:
# #             break

# #         datagrams += 1
# #         bytes_received += len(data)

# #         if outfile:
# #             outfile.write(data)

# #         # Check TS sync bytes (188-byte packets)
# #         if len(data) % 188 == 0:
# #             for i in range(0, len(data), 188):
# #                 if data[i] != 0x47:
# #                     bad_packets += 1
# #                     break

# #         now = time.time()

# #         if now - last >= 1:
# #             elapsed = now - start
# #             bitrate = (bytes_received * 8) / elapsed / 1e6

# #             print(
# #                 f"[{elapsed:7.1f}s] "
# #                 f"{datagrams:8,d} datagrams  "
# #                 f"{bytes_received/1024/1024:8.2f} MB  "
# #                 f"{bitrate:6.2f} Mbps  "
# #                 f"Bad TS: {bad_packets}"
# #             )

# #             last = now

# #     print("\nStopped.")
# #     print(f"Datagrams : {datagrams:,}")
# #     print(f"Bytes     : {bytes_received:,}")
# #     print(f"Bad TS    : {bad_packets}")

# #     if outfile:
# #         outfile.close()

# #     sock.close()


# # if __name__ == "__main__":
# #     main()

# import av
# import cv2

# container = av.open("udp://127.0.0.1:5601")

# print(container.streams)

# video_stream = None
# data_stream = None

# for s in container.streams:
#     print(s.index, s.type, s.codec_context.name)

#     if s.type == "video":
#         video_stream = s

#     if s.type == "data":
#         data_stream = s

# for packet in container.demux():

#     if packet.stream.type == "video":

#         for frame in packet.decode():

#             img = frame.to_ndarray(format="bgr24")
#             cv2.imshow("Video", img)

#             if cv2.waitKey(1) == 27:
#                 break

#     elif packet.stream.type == "data":

#         print(packet)

import av
import cv2
import time
import threading
import queue
from collections import deque

UDP_URL = "udp://127.0.0.1:5601?fifo_size=1000000&overrun_nonfatal=1"

# Optional KLV decoder
try:
    from klvdata.streamparser import StreamParser
    HAVE_KLV = True
except Exception:
    HAVE_KLV = False


latest_metadata = {}
metadata_lock = threading.Lock()


def metadata_worker(data_queue):
    """
    Parses incoming KLV packets.
    Falls back to raw hex if klvdata cannot decode them.
    """

    parser = StreamParser() if HAVE_KLV else None

    while True:
        data = data_queue.get()

        if data is None:
            break

        try:

            if HAVE_KLV:

                parser.parse(data)

                decoded = {}

                for item in parser:
                    decoded[str(item.key)] = str(item.value)

                with metadata_lock:
                    latest_metadata.clear()
                    latest_metadata.update(decoded)

            else:

                with metadata_lock:
                    latest_metadata.clear()
                    latest_metadata["Raw KLV"] = data.hex()[:120]

        except Exception as e:

            with metadata_lock:
                latest_metadata.clear()
                latest_metadata["Parser Error"] = str(e)


def main():

    print("Opening stream...")
    container = av.open(
        UDP_URL,
        options={
            "fflags": "nobuffer",
            "flags": "low_delay"
        }
    )

    video_stream = None
    data_stream = None

    print("\nStreams found:\n")

    for s in container.streams:

        codec = getattr(s.codec_context, "name", None)

        print(
            f"Index={s.index:2d} "
            f"Type={s.type:6s} "
            f"Codec={codec}"
        )

        if s.type == "video":
            video_stream = s

        elif s.type == "data":
            data_stream = s

    if video_stream is None:
        raise RuntimeError("No video stream found.")

    data_queue = queue.Queue()

    if data_stream is not None:
        threading.Thread(
            target=metadata_worker,
            args=(data_queue,),
            daemon=True
        ).start()

    frame_counter = 0
    fps_timer = time.time()
    fps = 0

    while True:

        for packet in container.demux():

            ############################################
            # VIDEO
            ############################################

            if packet.stream.type == "video":

                for frame in packet.decode():

                    img = frame.to_ndarray(format="bgr24")

                    frame_counter += 1

                    if time.time() - fps_timer >= 1:

                        fps = frame_counter
                        frame_counter = 0
                        fps_timer = time.time()

                    cv2.putText(
                        img,
                        f"FPS : {fps}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2
                    )

                    y = 60

                    with metadata_lock:

                        for k, v in latest_metadata.items():

                            text = f"{k}: {v}"

                            cv2.putText(
                                img,
                                text[:120],
                                (15, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,255,255),
                                1
                            )

                            y += 22

                            if y > img.shape[0]-20:
                                break

                    cv2.imshow("Video + Metadata", img)

                    key = cv2.waitKey(1)

                    if key == 27:
                        return

            ############################################
            # KLV
            ############################################

            elif packet.stream.type == "data":

                try:
                    data_queue.put(packet.to_bytes())
                except Exception:
                    pass


if __name__ == "__main__":
    main()