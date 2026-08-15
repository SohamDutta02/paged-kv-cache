# paged-kv-server demo image.
#
# This never touches paged-kv-cuda: the server crate depends only on
# paged-kv-core (pure scheduling logic, no device backend). That means this
# image needs no CUDA toolchain, no GPU, nothing beyond a standard Rust
# toolchain to build and a bare Linux base to run — the whole point of the
# systems demo being usefully deployable anywhere, for free, today.

# ---- build stage ----
FROM rust:slim-bookworm AS builder
WORKDIR /app

# Whole workspace at once — three small crates, not worth the
# dependency-caching-layer trick (dummy src files + prebuild) that pays off
# on much larger projects with slow rebuild cycles.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Only paged-kv-server actually gets compiled. Cargo resolves the full
# workspace lockfile but only builds the dependency subgraph this binary
# actually needs, so cudarc and its deps are never touched here even if
# they're present in the lockfile from an unrelated `--features cuda` build.
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