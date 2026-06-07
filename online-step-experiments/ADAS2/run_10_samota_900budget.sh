#!/bin/bash
# Run 10 PFES+SAMOTA experiments with 900 budget (matching PFES)
# Saves results in results_10runs_samota_900budget/ with detailed logs

OUTPUT_DIR="results_10runs_samota_900budget"
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "RUNNING 10 PFES+SAMOTA RUNS (900 BUDGET, NEW VERSION)"
echo "========================================================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

for run in {1..10}; do
    echo ""
    echo "========================================================================"
    echo "RUN $run/10 - Starting at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"

    # Clean up old results
    rm -rf pfes_samota_baseline

    # Run PFES+SAMOTA
    echo "Executing: python3 PFES_SAMOTA.py"
    python3 PFES_SAMOTA.py

    # Save results to run-specific directory
    if [ -d "pfes_samota_baseline" ]; then
        RUN_DIR="$OUTPUT_DIR/run_$run"
        mkdir -p "$RUN_DIR"

        # Copy all files (CSV, logs, etc.)
        cp -r pfes_samota_baseline/* "$RUN_DIR/"

        # Extract key metrics
        if [ -f "$RUN_DIR/reqs_NSGA3_1.csv" ]; then
            VIOLATIONS=$(head -2 "$RUN_DIR/reqs_NSGA3_1.csv" | tail -1 | cut -d',' -f4)
            echo "✓ Run $run complete: $VIOLATIONS violations found"
            echo "  Results saved to: $RUN_DIR"
        else
            echo "✓ Run $run complete"
            echo "  Results saved to: $RUN_DIR"
        fi

        # Clean up the baseline directory for next iteration
        rm -rf pfes_samota_baseline
    else
        echo "✗ Run $run FAILED - no output directory"
    fi

    echo "  Completed at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "========================================================================"
echo "✓ ALL 10 RUNS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: $OUTPUT_DIR/"
echo ""
echo "Directory structure:"
ls -lh "$OUTPUT_DIR/" | tail -15
echo ""
echo "To analyze results, run:"
echo "  python3 compare_10runs.py --pfes results_10runs_pfes --samota $OUTPUT_DIR"
echo "========================================================================"
