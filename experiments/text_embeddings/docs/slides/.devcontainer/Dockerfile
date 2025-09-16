FROM jetpackio/devbox:latest

# Installing your devbox project
WORKDIR /code
USER root:root
RUN mkdir -p /code && chown ${DEVBOX_USER}:${DEVBOX_USER} /code
USER ${DEVBOX_USER}:${DEVBOX_USER}
COPY --chown=${DEVBOX_USER}:${DEVBOX_USER} devbox.json devbox.json
COPY --chown=${DEVBOX_USER}:${DEVBOX_USER} devbox.lock devbox.lock
COPY --chown=${DEVBOX_USER}:${DEVBOX_USER} flake.nix flake.nix
COPY --chown=${DEVBOX_USER}:${DEVBOX_USER} flake.lock flake.lock

RUN devbox run -- echo "Installed Packages." && nix-store --gc 

RUN devbox shellenv --init-hook >> ~/.profile
