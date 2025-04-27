#!/bin/bash

while true; do
	scancel $(squeue | cut -c 11- | cut -c -10 | tail -n 1)
done
