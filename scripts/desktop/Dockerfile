FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libgtk-3-dev libwebkit2gtk-4.1-dev \
    libayatana-appindicator3-dev librsvg2-dev patchelf libdbus-1-dev \
    ca-certificates xvfb xauth dbus-x11 webkit2gtk-driver && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
