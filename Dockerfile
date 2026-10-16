# Kept in a container (used for the public demo link, e.g. on Render's free plan).
# NVIDIA Nemotron models are called on Nebius Token Factory: set NEBIUS_API_KEY in the host's
# environment settings, never in this file. ffmpeg takes the frames out of app-history videos.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH KEPT_PUBLIC=1 KEPT_DATA_DIR=/tmp/kept-data
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt gunicorn
COPY --chown=user . .
EXPOSE 10000
# Hosting services give the port in $PORT (Render uses 10000).
CMD gunicorn -w 1 --threads 8 --timeout 180 -b 0.0.0.0:${PORT:-10000} wsgi:app
