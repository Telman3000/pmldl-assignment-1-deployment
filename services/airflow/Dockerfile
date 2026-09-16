FROM apache/airflow:2.9.3-python3.11

USER root

# Install Docker CLI + Compose plugin (NOT docker.io — that has no `docker compose`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && . /etc/os-release \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian ${VERSION_CODENAME} stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && docker --version \
    && docker compose version

# Pipeline Python deps (baked in; avoid reinstalling on every container start).
USER airflow
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    mlflow==2.19.0 \
    joblib==1.4.2

# Root is needed to talk to the mounted docker.sock from the host.
USER root
