import warnings
from street_network_processor import StreetNetworkProcessor
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
## USER INPUT
# --------------------------------------------------------------------------

# Define the list of buffer distances to analyze (in miles)
buffer_list_mile = [0.25, 0.5, 0.75, 1.0]

# Define the list of your control variables located in the attributes of neighborhood shapefile
ctrl_vars = ['bachdg', 'emp', 'home_own', 'resoccup', 'sinfamrate', 'sinfamage']

# --------------------------------------------------------------------------
## Configuration
# --------------------------------------------------------------------------

source_edges_path = "../data/streetnetwork/streets.shp"
NPA_shape_path = "../data/neighborhoods/neighborhoods.shp"

# --------------------------------------------------------------------------
## Execution Block
# --------------------------------------------------------------------------
if __name__ == "__main__":

    print("Starting Street Network Analysis Pipeline.")

    print("\nInitializing Street Network Processor...")
    # Initialize the processor with configuration paths
    processor = StreetNetworkProcessor(source_edges_path, NPA_shape_path)

    try:
        # 1. Extract nodes and index edges for the entire network
        processor.extract_nodes()

        # 2. Simplify the global network by removing redundant degree-2 nodes
        processor.simplify_network()

        # 3-5. Process local networks for all defined buffers, calculate measures, and merge results
        processor.process_buffers(buffer_list_mile)

        print("\n✅ Network Processing Complete. Final dataset ready for analysis.")

    except Exception as e:
        print(f"\n❌ An error occurred during processing: {e}")
