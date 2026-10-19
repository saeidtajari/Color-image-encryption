#!/bin/bash
# usage: STS_DIR=/path/to/NIST-Statistical-Test-Suite/sts sts_run.sh name file nbits nstreams
# Runs the NIST SP 800-22 suite (built separately) on a binary file and copies the report to results/sts_<name>.txt.
set -e
name=$1; file=$(realpath "$2"); nbits=$3; ns=$4
src=${STS_DIR:?set STS_DIR to the directory of the built NIST suite (the one that contains ./assess)}
wd=${STS_WORK:-/tmp/sts}/run_$name
out=$(pwd)/results/sts_$name.txt
rm -rf "$wd"; mkdir -p "$(dirname "$wd")"; cp -r "$src" "$wd"; cd "$wd"
printf "0\n$file\n1\n0\n$ns\n1\n" | ./assess $nbits > run.log 2>&1 || true   # the suite does not return 0
cp experiments/AlgorithmTesting/finalAnalysisReport.txt "$out"
echo done $name
