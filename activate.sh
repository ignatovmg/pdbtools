#!/usr/bin/env bash

ROOT="$(cd "$(dirname ${BASH_SOURCE})" && pwd)"
export PATH="${ROOT}/venv/bin:${PATH}"
conda activate ${ROOT}/venv
