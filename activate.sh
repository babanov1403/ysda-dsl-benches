#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    case ${0##*/} in
        activate.sh) exec bash "$0" "$@" ;;
        *)
            echo "Run 'sh ./activate.sh' to open an activated Bash shell." >&2
            return 1
            ;;
    esac
fi

if [[ ! -r "$(dirname -- "${BASH_SOURCE[0]}")/.venv/bin/activate" ]]; then
    echo "From the repository root, run 'UV_PROJECT_ENVIRONMENT=.venv uv sync --locked' first." >&2
    return 1 2>/dev/null || exit 1
fi

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    exec bash --rcfile "$0" -i
fi

source "$(dirname -- "${BASH_SOURCE[0]}")/.venv/bin/activate" || return
export UV_PROJECT_ENVIRONMENT=.venv
