#!/bin/bash
# Levi build monitor — live progress bar with ETA

while true; do
    LINE=$(tmux capture-pane -t levi -p 2>/dev/null | grep -oP '\d+/\d+ \(\d+%\)' | tail -1)
    if [ -z "$LINE" ]; then
        echo "No build running in tmux session 'levi'"
        exit 1
    fi

    CURRENT=$(echo "$LINE" | grep -oP '^\d+')
    TOTAL=$(echo "$LINE" | grep -oP '/\K\d+')
    PCT=$(echo "$LINE" | grep -oP '\(\K\d+')

    # Track rate over last two samples (pure bash, no bc)
    if [ -n "$PREV_CURRENT" ] && [ -n "$PREV_TIME" ]; then
        NOW=$(date +%s)
        ELAPSED=$((NOW - PREV_TIME))
        DELTA=$((CURRENT - PREV_CURRENT))
        if [ "$ELAPSED" -gt 0 ] && [ "$DELTA" -gt 0 ]; then
            RATE=$((DELTA * 10 / ELAPSED))
            RATE_STR="$((RATE / 10)).$((RATE % 10)) v/s"
            REMAINING=$((TOTAL - CURRENT))
            ETA_SECS=$((REMAINING * ELAPSED / DELTA))
            ETA_MIN=$((ETA_SECS / 60))
            ETA_HR=$((ETA_MIN / 60))
            ETA_MIN_R=$((ETA_MIN % 60))
            if [ "$ETA_HR" -gt 0 ]; then
                ETA_STR="${ETA_HR}h ${ETA_MIN_R}m"
            else
                ETA_STR="${ETA_MIN}m"
            fi
        fi
    fi
    PREV_CURRENT=$CURRENT
    PREV_TIME=$(date +%s)

    # Progress bar
    BAR_WIDTH=40
    FILLED=$((PCT * BAR_WIDTH / 100))
    EMPTY=$((BAR_WIDTH - FILLED))
    BAR=$(printf '%0.s█' $(seq 1 $FILLED 2>/dev/null))
    SPACE=$(printf '%0.s░' $(seq 1 $EMPTY 2>/dev/null))

    printf "\r\033[K [$BAR$SPACE] %d/%d (%d%%) | %s | ETA: %s " \
        "$CURRENT" "$TOTAL" "$PCT" "${RATE_STR:-...}" "${ETA_STR:-calculating...}"

    sleep 5
done
