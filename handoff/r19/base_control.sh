#!/usr/bin/env bash
set -u
C=$HOME/ember-ws/checkouts/a3a9bf32763b98486a2ab95467ae4599ab787349
export PYTHONPATH=$C/src
export HF_HOME=$HOME/ember-ws/hf-home
export EMBER_CHECKOUT=$C
export TOKENIZERS_PARALLELISM=false
export EMBER_BLOCK=4
cd "$C"
exec "$HOME/ember-ws/venvs/9613f929b53c0fed/bin/python" \
  /mnt/c/Users/Admin/Documents/Test/ws-tmp/run_base_control.py
