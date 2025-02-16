#!/bin/bash

# This script checks if specified columns (x, y, z) contain valid float values in a .4dn or .ecsv file.
# Usage: ./verify_floats.sh input_file.4dn

FILE=$1

if [[ ! -f "$FILE" ]]; then
    echo "Error: File not found!"
    exit 1
fi

# Extract the first non-comment line (assumed to be the header)
HEADER=$(grep -v '^#' "$FILE" | head -n 1)

# Print the detected header
echo "Detected header: $HEADER"

# Identify column positions (handling both space and comma delimiters)
X_COL=$(echo "$HEADER" | awk -F'[ ,]+' '{for(i=1;i<=NF;i++) if($i=="x" || $i=="X") print i}')
Y_COL=$(echo "$HEADER" | awk -F'[ ,]+' '{for(i=1;i<=NF;i++) if($i=="y" || $i=="Y") print i}')
Z_COL=$(echo "$HEADER" | awk -F'[ ,]+' '{for(i=1;i<=NF;i++) if($i=="z" || $i=="Z") print i}')

if [[ -z "$X_COL" || -z "$Y_COL" || -z "$Z_COL" ]]; then
    echo "Error: x, y, or z columns not found in header."
    exit 1
fi

# Check for invalid float values in x, y, z columns
echo "Checking for non-float values..."
awk -F'[ ,]+' -v x=$X_COL -v y=$Y_COL -v z=$Z_COL '{
    if (NR > 1 && !/^#/) {
        if ($x !~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) print "Invalid value in x column at line " NR ": " $0;
        if ($y !~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) print "Invalid value in y column at line " NR ": " $0;
        if ($z !~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) print "Invalid value in z column at line " NR ": " $0;
    }
}' "$FILE"

echo "Validation complete."
