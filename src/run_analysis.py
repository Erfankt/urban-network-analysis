"""
Stage 1: build the neighborhood street networks and measure their configuration.

    python run_analysis.py              interactive: every input is prompted for, Enter accepts the default
    python run_analysis.py --reuse      rerun with the previously saved answers, without prompts
                                        (also the way to continue an interrupted run)
    python run_analysis.py --streets ../data/streetnetwork/streets.shp \\
                           --neighborhoods ../data/neighborhoods/neighborhoods.shp --id-col id \\
                           --buffers 0.1,0.25,0.5,0.75,1.0 --crime propcrimct --controls bachdg,med_age,emp
                                        answers given up front; anything missing is prompted for

Inputs and choices are recorded in data/final_dataset/user_selection.json, which stage 2
(pca_analysis.ipynb / pca_analysis.py) reads. Output: data/final_dataset/extracted_measures.shp,
the neighborhoods layer with one column per measure and buffer.

The measures are checkpointed neighborhood by neighborhood in data/checkpoints/ (one CSV per
buffer), so a run that stops resumes where it was and buffers already measured for the same
input layers are reused; --recompute measures them again.
"""
import argparse, warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
## DEFAULTS offered by the prompts (press Enter to accept)
# --------------------------------------------------------------------------
DEFAULTS = {
    "streets_path": "../data/streetnetwork/streets.shp",
    "neighborhoods_path": "../data/neighborhoods/neighborhoods.shp",
    "buffer_list_mile": [0.1, 0.25, 0.5, 0.75, 1.0],   # the five nested bands of the paper (Section 3.3.3)
}

# --------------------------------------------------------------------------
## Execution Block
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--streets", metavar="PATH", help="street centerline shapefile")
    parser.add_argument("--neighborhoods", metavar="PATH", help="neighborhood polygons shapefile")
    parser.add_argument("--id-col", metavar="COLUMN", help="column identifying each neighborhood (default: id)")
    parser.add_argument("--buffers", metavar="MILES", help="comma-separated distances, e.g. 0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--crime", metavar="COLUMN", help="dependent variable (crime count column)")
    parser.add_argument("--controls", metavar="COLUMNS", help="comma-separated control variable columns")
    parser.add_argument("--reuse", action="store_true", help="reuse the saved selection without asking")
    parser.add_argument("--recompute", action="store_true",
                        help="ignore the checkpoints and measure every buffer of this run again")
    args = parser.parse_args()

    from user_inputs import run_setup, id_column
    from street_network_processor import StreetNetworkProcessor

    print("Starting Street Network Analysis Pipeline.")
    selection = run_setup(
        DEFAULTS,
        preset={"streets_path": args.streets, "neighborhoods_path": args.neighborhoods, "id_col": args.id_col,
                "buffers": args.buffers, "crime_col": args.crime, "ctrl_vars": args.controls},
        reuse=args.reuse,
    )

    print("\nInitializing Street Network Processor...")
    processor = StreetNetworkProcessor(selection["streets_path"], selection["neighborhoods_path"],
                                       id_col=id_column(selection))

    try:
        processor.extract_nodes()
        processor.simplify_network()
        processor.process_buffers(selection["buffer_list_mile"], recompute=args.recompute)

        print("\n✅ Network Processing Complete. Final dataset ready for analysis.")

    except KeyboardInterrupt:
        print("\nInterrupted. The neighborhoods measured so far are checkpointed; "
              "run `python run_analysis.py --reuse` to continue.")
    except Exception as e:
        print(f"\n❌ An error occurred during processing: {e}")
        print("The neighborhoods measured so far are checkpointed; once the cause is fixed, "
              "run `python run_analysis.py --reuse` to continue.")
