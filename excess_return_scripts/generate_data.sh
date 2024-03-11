#!/usr/bin/env bash

set -eu -o pipefail
set -xv

echo "Script expects to be in folder that contains PDFs of earning call transcripts" 1>&2
for x in *.pdf; do pdftotext "$x"; done
bash -xv ./extract_date.sh > stock_quarterly_call_dates.csv
python3 download_stock_date.py stock_quarterly_call_dates.csv data_output.csv return_output.csv
