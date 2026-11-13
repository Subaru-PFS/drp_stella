#!/bin/bash
# Claude: run mergeArms and coaddSpectra from a drp_stella checkout, for a regression check
#
# Usage: pfsRunRegression.sh -c CHECKOUT -i INPUT -o PREFIX [-m MERGE_QUERY] [-a COADD_QUERY] [-j JOBS] [-b REPO]
#
#   CHECKOUT     drp_stella checkout to set up (built); don't edit it while this runs
#   INPUT        input collection (e.g. a weekly reduction, with pfsArm and fluxCal)
#   PREFIX       output collections are PREFIX-merge and PREFIX-coadd
#   MERGE_QUERY  data query for mergeArms (default: two hilo visits of the 2026.04 weekly)
#   COADD_QUERY  data query for coaddSpectra (default: one combination of that weekly)
#
# Then compare two runs with, e.g.:
#   pfsCompareRuns.py REPO REF-merge NEW-merge --datasets pfsMerged --noise
#   pfsCompareRuns.py REPO REF-coadd NEW-coadd --datasets pfsCoadd --noise
#
# PYTHONHASHSEED is fixed (override with HASHSEED=random), so that any code that still iterates over a
# set does so reproducibly.

set -o pipefail
REPO=/work/datastore
MERGE_QUERY="instrument='PFS' AND visit IN (122771, 122783)"
COADD_QUERY="instrument='PFS' AND combination='weekly_hilo/20260201'"
JOBS=8
while getopts "c:i:o:m:a:j:b:" opt; do
    case $opt in
        c) CHECKOUT=$OPTARG ;;
        i) INPUT=$OPTARG ;;
        o) PREFIX=$OPTARG ;;
        m) MERGE_QUERY=$OPTARG ;;
        a) COADD_QUERY=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        b) REPO=$OPTARG ;;
        *) sed -n 2,17p "$0"; exit 1 ;;
    esac
done
if [ -z "$CHECKOUT" ] || [ -z "$INPUT" ] || [ -z "$PREFIX" ]; then
    sed -n 2,17p "$0"
    exit 1
fi

setup -k -r "$CHECKOUT" -j
export PYTHONHASHSEED=${HASHSEED:-0}
LOGS=$(mktemp -d)
echo "drp_stella: $(git -C "$CHECKOUT" log --oneline -1) ($(git -C "$CHECKOUT" status --short | wc -l) uncommitted)"

pipetask run -b "$REPO" -i "$INPUT" -o "$PREFIX-merge" -j "$JOBS" \
    -p "$DRP_STELLA_DIR/pipelines/reduceExposure.yaml#mergeArms" -d "$MERGE_QUERY" > "$LOGS/merge.log" 2>&1
echo "merge: $(grep 'Executed\|ERROR' "$LOGS/merge.log" | tail -1)"
pipetask run -b "$REPO" -i "$INPUT" -o "$PREFIX-coadd" -j "$JOBS" \
    -p "$DRP_STELLA_DIR/pipelines/science.yaml#coaddSpectra" -d "$COADD_QUERY" > "$LOGS/coadd.log" 2>&1
echo "coadd: $(grep 'Executed\|ERROR' "$LOGS/coadd.log" | tail -1)"
echo "logs in $LOGS"
