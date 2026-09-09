
FROM rust:slim-bookworm AS builder
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates


RUN cargo build --release -p paged-kv-server

# ---- runtime stage ----
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin demo
WORKDIR /app

COPY --from=builder /app/target/release/paged-kv-server ./paged-kv-server
COPY crates/paged-kv-server/static ./static

ENV STATIC_DIR=/app/static
USER demo

EXPOSE 8080
CMD ["./paged-kv-server"]
