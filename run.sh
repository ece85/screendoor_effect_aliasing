#!/bin/bash
chmod +x setup_venv.sh
./setup_venv.sh
source .venv/bin/activate
python simulate_screen_door.py
