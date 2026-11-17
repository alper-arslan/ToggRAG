FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --extra ui --no-install-project

COPY src ./src
COPY config.yaml ./
COPY eval ./eval
RUN uv sync --locked --no-dev --extra ui

# The manual and the index live in ./data, mounted at run time (see docker-compose.yml).
ENV EMBEDDING__BASE_URL=http://host.docker.internal:11434 \
    LLM__BASE_URL=http://host.docker.internal:11434/v1

EXPOSE 7860
CMD ["uv", "run", "--no-sync", "toggrag", "ui", "--host", "0.0.0.0"]
