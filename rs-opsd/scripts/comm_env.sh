#!/usr/bin/env bash
# Source before Python/Ray starts. By default leave the communication library
# environment untouched. An administrator can opt into a library-specific
# statistics workaround by supplying its documented variable name and value.
if [[ "${RS_OPSD_COMM_STATS_MODE:-inherit}" != inherit ]]; then
  if [[ ! "${RS_OPSD_COMM_STATS_ENV:-}" =~ ^[A-Z][A-Z0-9_]*_STATS_MODE$ ]]; then
    echo "ERROR: set RS_OPSD_COMM_STATS_ENV to the library's *_STATS_MODE variable" >&2
    return 2
  fi
  export "${RS_OPSD_COMM_STATS_ENV}=${RS_OPSD_COMM_STATS_MODE}"
  echo "Communication statistics override: ${RS_OPSD_COMM_STATS_ENV}=${RS_OPSD_COMM_STATS_MODE}"
fi
