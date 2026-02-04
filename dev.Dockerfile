ARG PLATFORM=linux/amd64

FROM --platform=${PLATFORM} ghcr.io/astral-sh/uv:python3.10-trixie

COPY . .

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv sync --all-groups
     && uv pip install ipykernel

CMD ["uv", "run", "python"]
