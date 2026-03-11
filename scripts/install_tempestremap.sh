#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  install_tempestremap.sh conda
  install_tempestremap.sh source [PREFIX]

Modes:
  conda   Install TempestRemap from conda-forge into the active conda environment.
  source  Build TempestRemap from source into PREFIX (default: $HOME/.local/tempestremap).
EOF
}

verify_bins() {
  for exe in GenerateCSMesh GenerateRLLMesh GenerateOverlapMesh GenerateOfflineMap ApplyOfflineMap; do
    if ! command -v "${exe}" >/dev/null 2>&1; then
      echo "Missing executable: ${exe}" >&2
      exit 1
    fi
  done
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

mode="$1"
case "${mode}" in
  conda)
    conda install -c conda-forge tempest-remap
    verify_bins
    ;;
  source)
    prefix="${2:-$HOME/.local/tempestremap}"
    workdir="$(mktemp -d)"
    trap 'rm -rf "${workdir}"' EXIT
    git clone https://github.com/ClimateGlobalChange/tempestremap.git "${workdir}/tempestremap"
    cd "${workdir}/tempestremap"
    if [[ -x ./autogen.sh ]]; then
      ./autogen.sh
    fi
    ./configure --prefix="${prefix}"
    make -j"$(nproc)"
    make install
    export PATH="${prefix}/bin:${PATH}"
    verify_bins
    echo "Add ${prefix}/bin to PATH to use TempestRemap outside this shell."
    ;;
  *)
    usage
    exit 1
    ;;
esac
