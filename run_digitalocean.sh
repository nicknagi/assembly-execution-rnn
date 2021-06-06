#!/bin/bash

export GLOO_SOCKET_IFNAME=eth1 && \
python main.py --enable-ddp --world-size $1 --machine-num $2
