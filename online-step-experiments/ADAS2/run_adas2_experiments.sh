#!/bin/bash
# Run 10 PFES + 10 PFES+SAMOTA experiments for ADAS2 (900 budget each)

echo "========================================================================"
echo "ADAS2 EXPERIMENTS: 10 PFES + 10 PFES+SAMOTA RUNS"
echo "========================================================================"
echo ""

# Run 10 PFES baseline runs
echo "========================================================================"
echo "PHASE 1: Running 10 PFES baseline runs (900 evaluations each)"
echo "========================================================================"
echo ""

PFES_DIR="results_10runs_pfes"
mkdir -p "$PFES_DIR"

for run in {1..10}; do
    echo "PFES Run $run/10 - $(date '+%H:%M:%S')"
    mkdir -p "$PFES_DIR/run_$run"

    python3 PFES_falsification.py \
        --size 30 \
        --niterations 30 \
        --nruns 1 \
        --optalg NSGA3 \
        --logdir "$PFES_DIR/run_$run" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        # Extract violations count
        if [ -f "$PFES_DIR/run_$run/reqs_NSGA3_1.csv" ]; then
            VIOLATIONS=$(head -2 "$PFES_DIR/run_$run/reqs_NSGA3_1.csv" | tail -1 | cut -d',' -f4)
            echo "  ✓ Complete: $VIOLATIONS violations"
        else
            echo "  ✓ Complete"
        fi
    else
        echo "  ✗ Failed"
    fi
done

echo ""
echo "========================================================================"
echo "PHASE 2: Running 10 PFES+SAMOTA runs (900 evaluations each)"
echo "========================================================================"
echo ""

SAMOTA_DIR="results_10runs_samota_900budget"
mkdir -p "$SAMOTA_DIR"

for run in {1..10}; do
    echo "PFES+SAMOTA Run $run/10 - $(date '+%H:%M:%S')"

    # Clean up old results
    rm -rf pfes_samota_baseline

    # Run PFES+SAMOTA
    python3 PFES_SAMOTA.py > /dev/null 2>&1

    # Save results
    if [ -d "pfes_samota_baseline" ]; then
        RUN_DIR="$SAMOTA_DIR/run_$run"
        mkdir -p "$RUN_DIR"
        cp -r pfes_samota_baseline/* "$RUN_DIR/"

        if [ -f "$RUN_DIR/reqs_NSGA3_1.csv" ]; then
            VIOLATIONS=$(head -2 "$RUN_DIR/reqs_NSGA3_1.csv" | tail -1 | cut -d',' -f4)
            EVALS=$(wc -l < "$RUN_DIR/F_all_evaluations_NSGA3_0.csv")
            echo "  ✓ Complete: $EVALS evals, $VIOLATIONS violations"
        else
            echo "  ✓ Complete"
        fi

        rm -rf pfes_samota_baseline
    else
        echo "  ✗ Failed"
    fi
done

echo ""
echo "========================================================================"
echo "✓ ALL EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results directories:"
echo "  PFES:         $PFES_DIR/"
echo "  PFES+SAMOTA:  $SAMOTA_DIR/"
echo ""
echo "To compare results:"
echo "  python3 compare_10runs.py"
echo ""
echo "To generate plots:"
echo "  python3 plot_comparison.py"
echo ""
echo "========================================================================"
