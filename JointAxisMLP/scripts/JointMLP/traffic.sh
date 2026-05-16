#!/bin/bash
set -e
DATASET_KEY=traffic bash "$(dirname -- "${BASH_SOURCE[0]}")/_template.sh"
