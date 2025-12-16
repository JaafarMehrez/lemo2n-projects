#!/bin/bash

# Timesteps you found from the temperature analysis
TIMESTEPS=(35800 60400 183900 254200 291400)

echo "Extracting 5 frames from trajectory..."

for i in {1..5}; do
    TS=${TIMESTEPS[$((i-1))]}
    echo "Extracting Frame $i (timestep $TS)..."
    
    # Method 1: Using awk (most reliable)
    awk -v ts="$TS" '
    /ITEM: TIMESTEP/ {
        if (found) exit
        getline
        if ($0 == ts) {
            found = 1
            print "ITEM: TIMESTEP"
            print ts
            while (getline) {
                if (/ITEM: TIMESTEP/) exit
                print
            }
        }
    }' nvt_equil.lammpstrj > frame_${i}.dump
    
    # Check if extraction was successful
    if [ -s frame_${i}.dump ]; then
        ATOMS=$(grep -c "^[0-9]" frame_${i}.dump)
        echo "  ✓ Extracted frame_${i}.dump with $ATOMS atoms"
    else
        echo "  ✗ Failed to extract frame_${i}.dump"
    fi
done

echo ""
echo "Summary of extracted frames:"
for i in {1..5}; do
    if [ -f frame_${i}.dump ]; then
        LINES=$(wc -l < frame_${i}.dump)
        echo "  frame_${i}.dump: $LINES lines"
    fi
done
