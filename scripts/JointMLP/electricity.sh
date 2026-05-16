#!/bin/bash
set -e
DATASET_KEY=electricity bash "$(dirname -- "${BASH_SOURCE[0]}")/_template.sh"
