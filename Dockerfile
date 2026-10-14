FROM node:24-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM rust:slim-bookworm AS server
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --locked -p paged-kv-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /usr/sbin/nologin demo
WORKDIR /app
COPY --from=server /app/target/release/paged-kv-server ./paged-kv-server
COPY --from=web /web/dist ./static
ENV STATIC_DIR=/app/static
USER demo
EXPOSE 8080
CMD ["./paged-kv-server"]
