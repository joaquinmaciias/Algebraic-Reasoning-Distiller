FROM ubuntu:noble

# TO BE PARSED FROM docker-compose.yaml
ARG UID=
ARG GID=
ARG USERNAME=
ARG INSTALL_DOCKER=false


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        openssh-client \
        sudo \
        tmux \
        tree \
        build-essential \
        libgmp-dev \
        libcdd-dev \
        micro \
        gawk \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# install docker dependencies only if requested
RUN set -eux; \
    if [ "${INSTALL_DOCKER}" = "true" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            gnupg \
            lsb-release; \
        rm -rf /var/lib/apt/lists/*; \
        install -m 0755 -d /etc/apt/keyrings; \
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
          | gpg --dearmor -o /etc/apt/keyrings/docker.gpg; \
        chmod a+r /etc/apt/keyrings/docker.gpg; \
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
          https://download.docker.com/linux/ubuntu \
          $(. /etc/os-release && echo ${VERSION_CODENAME}) stable" \
          > /etc/apt/sources.list.d/docker.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
          docker-ce-cli docker-buildx-plugin docker-compose-plugin; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Create user/group with requested UID/GID, reusing existing GID if needed
RUN set -eux; \
    if getent group "${GID}" >/dev/null; then \
        EXISTING_GROUP="$(getent group "${GID}" | cut -d: -f1)"; \
        useradd --uid "${UID}" --gid "${GID}" -m -s /bin/bash "${USERNAME}"; \
        echo "Reused existing group ${EXISTING_GROUP} (GID=${GID})"; \
    else \
        groupadd --gid "${GID}" "${USERNAME}"; \
        useradd --uid "${UID}" --gid "${GID}" -m -s /bin/bash "${USERNAME}"; \
    fi; \
    usermod -aG sudo "${USERNAME}"; \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Home scaffolding
RUN mkdir -p "/home/${USERNAME}/.ssh" "/home/${USERNAME}/.config" "/home/${USERNAME}/.cache" \
    && chmod 700 "/home/${USERNAME}/.ssh" \
    && chown -R "${UID}:${GID}" "/home/${USERNAME}"
    
ENV DOCKER_CONFIG="/home/${USERNAME}/.docker"

USER ${USERNAME}


CMD ["bash", "-lc", "sleep infinity"]
