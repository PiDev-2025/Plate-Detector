#!/bin/bash
echo "Starting Tunisian Plate Detector API"
# Free up memory
echo "Cleaning memory..."
free -m
echo "Starting application..."
# Run with limited memory
python tunisian_plate_detector.py
