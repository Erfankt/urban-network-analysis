import warnings
import sys

warnings.filterwarnings("ignore")

RUNNING_FROM_R = "rpycall" in sys.modules

# --------------------------------------------------------------------------
## USER INPUT
# --------------------------------------------------------------------------

buffer_list_mile = [0.25, 0.5, 0.75, 1.0]
ctrl_vars = ['bachdg', 'emp', 'home_own', 'resoccup', 'sinfamrate', 'sinfamage']
crime_col = "propcrimct"

source_edges_path = "../data/streetnetwork/streets.shp"
NPA_shape_path = "../data/neighborhoods/neighborhoods.shp"

# --------------------------------------------------------------------------
## Execution Block
# --------------------------------------------------------------------------

if __name__ == "__main__" and not RUNNING_FROM_R:
    from street_network_processor import StreetNetworkProcessor

    print("Starting Street Network Analysis Pipeline.")
    print("\nInitializing Street Network Processor...")

    processor = StreetNetworkProcessor(source_edges_path, NPA_shape_path)

    try:
        processor.extract_nodes()
        processor.simplify_network()
        processor.process_buffers(buffer_list_mile)

        print("\n✅ Network Processing Complete. Final dataset ready for analysis.")

    except Exception as e:
        print(f"\n❌ An error occurred during processing: {e}")
