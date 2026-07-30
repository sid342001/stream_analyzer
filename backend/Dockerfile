# Build context must be the REPO ROOT, not backend/ — this image needs both
# model_servers/perception/ (SAM 3 + DINOv3, imported in-process) and
# uav_udp_stream/uavstream/ (KLV decode). See ../docker-compose.yml.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git build-essential libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# perception/ and uavstream/ are both already proper packages (__init__.py
# present) — copied in as-is so `from perception import ...` and
# `from uavstream.klv import ...` resolve as top-level imports.
COPY model_servers/perception/__init__.py model_servers/perception/interface.py /app/perception/
COPY uav_udp_stream/uavstream/ /app/uavstream/
COPY backend/app/ /app/app/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
