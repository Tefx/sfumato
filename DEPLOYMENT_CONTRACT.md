# Container Deployment Contract

## [Design] Scope

This artifact defines the pre-implementation contract for containerized deployment of `sfumato`.
It constrains the future `Dockerfile` and `docker-compose.yml` without committing this step to a
production-ready image or runtime layout.

## Boundary

- In scope: Docker build-stage contract, runtime requirements, compose service contract, volume and
  persistence semantics, startup and shutdown expectations.
- Out of scope: final package install commands, distro-specific package list, production hardening,
  image-size optimization, CI publishing, secrets management tooling.

## Design Basis

- [Proven] `ARCHITECTURE.md:1866` already defines container deployment as a single-process daemon
  running `sfumato watch` with persistent `/data` state.
- [Proven] `PROTOTYPING.md:62` establishes Playwright as mandatory for acceptable rendering quality,
  which makes browser runtime dependencies part of the container contract.
- [Proven] `README.md:42` positions the app as a long-lived daemon suitable for Docker hosts.

## Dockerfile Contract

### Required Stages

1. `base-runtime`
   - Responsibility: provide Python runtime plus shared OS-level libraries needed by Playwright,
     Pillow, and font rendering.
   - Must include: Python 3.12+ runtime, browser-compatible shared libraries, CJK-capable fonts,
     emoji-safe font fallback.
   - Must not include: app source layout assumptions that prevent reuse by later stages.

2. `builder`
   - Responsibility: assemble installable application artifacts from repo contents.
   - Inputs: `pyproject.toml`, lockfile if used by implementation, `src/`, `templates/`.
   - Output contract: the runtime stage can install or copy a fully built `sfumato` application
     without requiring editable installs.

3. `runtime`
   - Responsibility: host the executable daemon image.
   - Must contain:
     - the installed `sfumato` CLI entrypoint,
     - Playwright Chromium availability,
     - templates required by the renderer,
     - environment defaults for config and data paths.
   - Must expose these environment variables:
     - `SFUMATO_CONFIG=/data/config.toml`
     - `SFUMATO_DATA_DIR=/data`
     - `RIJKSMUSEUM_API_KEY` passthrough without baking secrets into the image.

### Runtime Requirements

- [Proven] Browser rendering support is mandatory; a runtime image that omits Playwright Chromium or
  equivalent browser assets violates the contract.
- [Likely] Font coverage must prioritize `Noto Sans/Serif CJK`-class families because the product
  renders multilingual news copy and prototype work validated CJK-heavy layouts.
- [Likely] The final runtime should run as a single application container with no sidecar process
  supervisor, because the architecture defines one long-lived daemon process.

### Failure Conditions

- Fails if the runtime image cannot execute `sfumato watch` directly.
- Fails if browser dependencies are deferred to host installation.
- Fails if secrets are copied into image layers instead of injected at runtime.
- Fails if the runtime stage writes durable state anywhere other than mounted paths under `/data`.

## Compose Service Contract

### Required Service Shape

- Service name: `sfumato`
- Lifecycle: long-running daemon, restart policy intended for unattended recovery.
- Command contract: start `sfumato watch` against the mounted config path.
- Stop contract: send `SIGTERM`, allow a bounded grace period for state flush and current-action
  completion.
- Health contract: future implementation must publish a freshness signal derived from the daemon's
  last successful loop or action.

### Required Compose Fields

- `build` or `image`: one must target the `runtime` stage contract.
- `environment`:
  - `SFUMATO_CONFIG=/data/config.toml`
  - `SFUMATO_DATA_DIR=/data`
  - `RIJKSMUSEUM_API_KEY` passthrough
- `volumes`: persistent mounts for config, paintings, state; optional retained mount for output.
- `stop_signal: SIGTERM`
- `stop_grace_period`: non-zero and long enough to finish a render/upload cycle.
- `healthcheck`: must evaluate daemon freshness, not just process existence.

### Networking Contract

- [Likely] `network_mode: host` is the default deployment contract on Linux Docker hosts because TV
  communication is local-network oriented and architecture guidance already assumes host-style
  reachability.
- [Speculative] macOS Docker Desktop deployments may require explicit IP routing instead of host
  networking. If that deployment target becomes primary, this contract should branch by platform
  rather than weakening Linux behavior.

### Healthcheck Contract

The production compose healthcheck must treat the container as healthy only when all of the
following are true:

- the daemon process is still running,
- the state directory is writable,
- a freshness artifact exists at `/data/state/last_action.json` or an equivalent contract path,
- the freshness artifact timestamp is newer than `2 * rotate_interval_minutes + startup_slack`.

This is intentionally stricter than `pgrep` because a wedged daemon loop is operationally dead even
if the PID survives.

## Persistence Semantics

### Mount Contract

Required durable paths under `/data`:

- `/data/config.toml`
  - Semantics: operator-authored configuration.
  - Persistence: durable.
  - Mutation policy: user-managed; container may read and validate but must not silently rewrite.

- `/data/paintings/`
  - Semantics: downloaded painting cache and sidecar metadata.
  - Persistence: durable.
  - Mutation policy: append/update by app; safe to retain across redeployments.

- `/data/state/`
  - Semantics: queue, used-painting marks, layout cache, embedding cache, health/freshness files.
  - Persistence: durable.
  - Mutation policy: application-owned, write-heavy, must survive container replacement.

- `/data/output/`
  - Semantics: rendered artifacts and local debugging output.
  - Persistence: optional but recommended for diagnostics.
  - Mutation policy: application-owned; may be pruned without corrupting durable state.

### Source-of-Truth Rules

- `config.toml` is the source of truth for operator intent.
- `/data/state/*` is the source of truth for resumable daemon progress.
- `/data/paintings/*` is the source of truth for the local art cache.
- `/data/output/*` is never a source of truth; it is disposable derivative output.

### Failure Conditions

- Fails if config and state are combined with ephemeral container filesystem only.
- Fails if output is treated as durable state required for restart correctness.
- Fails if image rebuilds invalidate caches that should have lived under mounts.

## Startup Entrypoint Contract

- Entrypoint responsibility: execute the application process directly, not via a long shell chain.
- Default startup mode: `sfumato watch --config /data/config.toml`.
- Preflight expectations:
  - config path existence is validated early,
  - mounted directories required for persistence exist or are created deterministically,
  - failures occur fast and loudly before the daemon enters its watch loop.
- Logging expectation: startup errors must reach container stdout/stderr.

## Graceful Stop Contract

- `SIGTERM` is the canonical shutdown signal.
- On shutdown, the daemon is expected to:
  1. stop accepting new work,
  2. finish or safely abort the current action boundary,
  3. flush in-memory state to `/data/state`,
  4. exit non-corruptly within the compose grace period.
- `SIGKILL` is treated as an orchestrator failure path, not a normal stop path.

## Trade-offs

- [Proven] Bundling browser dependencies increases image size, but prototype evidence shows quality
  loss is unacceptable without Playwright.
- [Likely] Host networking simplifies TV reachability, but reduces network isolation.
- [Likely] Durable mounts improve restart correctness, but require operators to manage filesystem
  ownership and backups.

## Open Questions

- What freshness artifact schema should the healthcheck read: timestamp only, or timestamp plus last
  action/result metadata?
- Is non-root execution a hard requirement for the production image, or only a later hardening goal?
- Should `/data/output` be mounted by default in compose, or enabled only for debugging profiles?

## Readiness

- Ready for implementation of a production Dockerfile and compose file.
- Not ready for image publishing or deployment automation until the open questions are resolved.
