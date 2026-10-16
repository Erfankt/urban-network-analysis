import warnings
import sys

warnings.filterwarnings("ignore")

RUNNING_FROM_R = "rpycall" in sys.modules

# --------------------------------------------------------------------------
## USER INPUT
# --------------------------------------------------------------------------

# Five nested buffer bands used in the paper (Section 3.3.3)
buffer_list_mile = [0.1, 0.25, 0.5, 0.75, 1.0]
# The dependent (crime count) column and the control variables are chosen
# interactively at run time (see user_inputs.py) and saved to
# data/final_dataset/user_selection.json for the notebook and R stages.

source_edges_path = "../data/streetnetwork/streets.shp"
NPA_shape_path = "../data/neighborhoods/neighborhoods.shp"

# --------------------------------------------------------------------------
## Execution Block
# --------------------------------------------------------------------------

if __name__ == "__main__" and not RUNNING_FROM_R:
    from street_network_processor import StreetNetworkProcessor

    print("Starting Street Network Analysis Pipeline.")
    print("\nInitializing Street Network Processor...")

    from user_inputs import select_variables
    crime_col, ctrl_vars = select_variables(NPA_shape_path)

    processor = StreetNetworkProcessor(source_edges_path, NPA_shape_path)

    try:
        processor.extract_nodes()
        processor.simplify_network()
        processor.process_buffers(buffer_list_mile)

        print("\n✅ Network Processing Complete. Final dataset ready for analysis.")

    except Exception as e:
        print(f"\n❌ An error occurred during processing: {e}")
