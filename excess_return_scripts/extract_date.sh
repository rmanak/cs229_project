#!/usr/bin/env bash
echo "Symbol,Date"
for x in *.txt; do
    symbol=$(echo "$x" | sed -E 's/.*\(([^)]*)\).*/\1/')
    date=$(awk '
       /^\x0c/ { exit;}
       /Transcript\)?$/ {
           read_date=1; next;
       }
       {
           if (read_date) {
               if (match($0, "^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\.? [0-9]+, [0-9]+")) {
                   print substr($0, RSTART, RLENGTH);
                }
                read_date=0
            }
        }' "$x")
    if [ -z "$date" ]; then
        echo "$x"
    fi
    printf "%s,%s\n" "$symbol" "$date"
 done
