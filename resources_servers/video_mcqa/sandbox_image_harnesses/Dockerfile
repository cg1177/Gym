# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sandbox image of the image-only agentic video configs (configs/*_agentic_image_only.yaml of videomme_v2, videomme_v1,
# lvbench and video_qa_rubric): the agentic video sandbox image (sandbox_image/Dockerfile) with the runtimes of Pi and
# Codex added. The lent sandbox reaches only the model server, so pi_agent and codex_agent cannot install their
# runtimes there; the configs point each adapter's sandbox_runtime_dir at the directory below, and the adapter checks
# at each seed that it runs the pinned version. OpenCode, FFmpeg, FFprobe and the video tools come from the base image
# unchanged, so its seed check still passes. FFmpeg and FFprobe stay in /usr/local/bin, on the PATH that /etc/profile
# sets for a login shell: Codex runs the model's commands with `bash -lc`.
#
# Each runtime has the layout that install_pi_runtime.sh and install_codex_runtime.sh produce (node/ and pi/, node/ and
# codex/), with the packages those installers fetch for Pi 0.80.2 and Codex 0.144.4 pinned by the lockfiles in pi/ and
# codex/. Both runtimes use one Node 22.19.0, the version both installers pin.
#
#   docker build --platform linux/amd64 --provenance=false \
#     --build-arg BASE_IMAGE=<image built from sandbox_image/Dockerfile> \
#     -t <registry>/video-mcqa-sandbox:<tag> resources_servers/video_mcqa/sandbox_image_harnesses

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# One layer: the context is mounted, not copied. Pi installs without package scripts, as its installer does.
RUN --mount=type=bind,target=/tmp/harness-runtimes-context \
    set -eu \
    && R=/opt/harness-runtimes \
    && curl -fsSL "https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-x64.tar.gz" -o /tmp/node.tar.gz \
    && echo "d36e56998220085782c0ca965f9d51b7726335aed2f5fc7321c6c0ad233aa96d  /tmp/node.tar.gz" | sha256sum -c - \
    && mkdir -p "$R/node" "$R/pi/pi" "$R/codex/codex" \
    && tar -xzf /tmp/node.tar.gz --strip-components=1 --no-same-owner -C "$R/node" \
    && rm /tmp/node.tar.gz \
    && ln -s ../node "$R/pi/node" \
    && ln -s ../node "$R/codex/node" \
    && cp /tmp/harness-runtimes-context/pi/package.json /tmp/harness-runtimes-context/pi/package-lock.json "$R/pi/pi/" \
    && cp /tmp/harness-runtimes-context/codex/package.json /tmp/harness-runtimes-context/codex/package-lock.json \
        "$R/codex/codex/" \
    && export PATH="$R/node/bin:$PATH" npm_config_cache=/tmp/harness-runtimes-npm \
    && (cd "$R/pi/pi" && npm ci --ignore-scripts --no-audit --no-fund) \
    && (cd "$R/codex/codex" && npm ci --no-audit --no-fund) \
    && rm -rf /tmp/harness-runtimes-npm /root/.npm \
    && test "$("$R/pi/node/bin/node" "$R/pi/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js" --version)" \
        = 0.80.2 \
    && mkdir /tmp/codex-version-home \
    && test "$(CODEX_HOME=/tmp/codex-version-home "$R/codex/node/bin/node" \
        "$R/codex/codex/node_modules/@openai/codex/bin/codex.js" --version)" = "codex-cli 0.144.4" \
    && rm -rf /tmp/codex-version-home \
    && touch "$R/pi/ready" "$R/codex/ready" \
    && bash -lc 'command -v ffmpeg ffprobe' > /dev/null
