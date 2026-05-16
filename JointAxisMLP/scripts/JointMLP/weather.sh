#!/bin/bash
set -e
DATASET_KEY=weather bash "$(dirname -- "${BASH_SOURCE[0]}")/_template.sh"
