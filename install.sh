#!/usr/bin/env bash
#
# Installer for the capburn_php native PHP extension.
#
# Downloads the prebuilt binary matching the current PHP x OS x arch from the
# eav93/capburn GitHub release, verifies the checksum, and prints how to run PHP
# with it.
#
# Quick install:
#   curl -fsSL https://raw.githubusercontent.com/eav93/capburn/main/install.sh | bash
#
# Options (flags or environment variables):
#   --dest DIR         where to put the .so (default ./capburn, or $CAPBURN_DEST)
#   --version vX.Y.Z   release version (default latest, or $CAPBURN_VERSION)
#   CAPBURN_PHP_BINARY  path to an existing .so (skips downloading)
#   GH_TOKEN / GITHUB_TOKEN  token for a private repository
#
# The path to the installed .so is printed to stdout (for use in scripts);
# progress and the run command go to stderr.
set -euo pipefail

REPO="eav93/capburn"
DEST="${CAPBURN_DEST:-./capburn}"
VERSION="${CAPBURN_VERSION:-latest}"

while [ $# -gt 0 ]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "capburn: unknown argument $1" >&2; exit 2 ;;
    esac
done

log() { echo "$@" >&2; }

# Prints the PHP run command to stderr.
print_usage() {
    local so="$1"
    log ""
    log "capburn: installed -> $so"
    log ""
    log "Run PHP with the extension:"
    log "  php -d extension=$so your-script.php"
    log ""
    log "Or enable it in php.ini (see \`php --ini\`):"
    log "  extension=$so"
    log ""
}

command -v php >/dev/null 2>&1 || { log "capburn: php not found in PATH"; exit 1; }

mkdir -p "$DEST"
OUT="$DEST/capburn_php.so"

# Explicit binary override — just copy it.
if [ -n "${CAPBURN_PHP_BINARY:-}" ] && [ -f "${CAPBURN_PHP_BINARY:-}" ]; then
    cp "$CAPBURN_PHP_BINARY" "$OUT"
    log "capburn: using CAPBURN_PHP_BINARY"
    ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
    print_usage "$ABS"
    echo "$ABS"
    exit 0
fi

PHP_VER="$(php -r 'echo PHP_MAJOR_VERSION.".".PHP_MINOR_VERSION;')"
case "$(uname -s)" in
    Linux)  OS=linux ;;
    Darwin) OS=macos ;;
    *) log "capburn: unsupported OS $(uname -s)"; exit 1 ;;
esac
case "$(uname -m)" in
    x86_64|amd64)  ARCH=x86_64 ;;
    arm64|aarch64) ARCH=aarch64 ;;
    *) log "capburn: unsupported architecture $(uname -m)"; exit 1 ;;
esac

# Authorization header for a private repository (when a token is set).
TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
auth_curl() {
    if [ -n "$TOKEN" ]; then
        curl -fsSL -H "Authorization: Bearer $TOKEN" "$@"
    else
        curl -fsSL "$@"
    fi
}

have_gh() { command -v gh >/dev/null 2>&1; }

# Resolve the version (latest -> newest release tag).
if [ "$VERSION" = latest ]; then
    if have_gh; then
        VERSION="$(gh release view -R "$REPO" --json tagName -q .tagName 2>/dev/null || true)"
    else
        VERSION="$(auth_curl "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
            | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/' || true)"
    fi
    [ -n "$VERSION" ] || { log "capburn: cannot resolve the latest version — pass --version"; exit 1; }
fi

ASSET="capburn_php-${VERSION}-php${PHP_VER}-${OS}-${ARCH}.so"
log "capburn: downloading ${ASSET} (${VERSION})"

# gh downloads reliably from a private repo too; curl works for public (or with a token).
if have_gh; then
    gh release download "$VERSION" -R "$REPO" -p "$ASSET" -O "$OUT" --clobber 2>/dev/null \
        && gh release download "$VERSION" -R "$REPO" -p "$ASSET.sha256" -O "$OUT.sha256" --clobber 2>/dev/null || true
else
    BASE="https://github.com/${REPO}/releases/download/${VERSION}"
    auth_curl "${BASE}/${ASSET}" -o "$OUT" || true
    auth_curl "${BASE}/${ASSET}.sha256" -o "$OUT.sha256" 2>/dev/null || true
fi

if [ ! -f "$OUT" ]; then
    log "capburn: failed to download ${ASSET}."
    log "capburn: a private repo needs gh or GH_TOKEN; a public one needs nothing."
    exit 1
fi

# Verify the checksum if it was downloaded.
if [ -f "$OUT.sha256" ]; then
    EXPECTED="$(awk '{print $1}' "$OUT.sha256")"
    if command -v shasum >/dev/null 2>&1; then
        ACTUAL="$(shasum -a 256 "$OUT" | awk '{print $1}')"
    elif command -v sha256sum >/dev/null 2>&1; then
        ACTUAL="$(sha256sum "$OUT" | awk '{print $1}')"
    else
        ACTUAL="$EXPECTED"
    fi
    rm -f "$OUT.sha256"
    if [ -n "$EXPECTED" ] && [ "$EXPECTED" != "$ACTUAL" ]; then
        log "capburn: checksum mismatch — file is corrupt"
        rm -f "$OUT"
        exit 1
    fi
fi

if [ ! -s "$OUT" ] || [ "$(wc -c < "$OUT")" -lt 4096 ]; then
    log "capburn: downloaded file is implausibly small"
    rm -f "$OUT"
    exit 1
fi

ABS="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
print_usage "$ABS"

echo "$ABS"
