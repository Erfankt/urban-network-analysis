import os, warnings, numpy as np, pandas as pd, geopandas as gpd, networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
from tqdm import tqdm
warnings.filterwarnings("ignore")


class StreetNetworkProcessor:
    """
    A unified class to manage the entire workflow for urban street network analysis:
    from initial data loading and cleaning, through network simplification, local
    network extraction based on buffers, to the final calculation and merging of
    graph-theoretic measures for Neighborhood Planning Areas (NPAs).
    """

    def __init__(self, source_edges_path, NPA_shape_path):
        """
        Initializes the processor by setting up file paths and loading the base data.
        """
        self.source_edges_path = source_edges_path
        self.NPA_shape_path = NPA_shape_path

        # Instance attributes to hold GeoDataFrames across processing steps
        self.source_edges = None  # Version 1 edges (with start/end node IDs)
        self.source_nodes = None  # Version 1 nodes
        self.edges_version_2 = None  # Simplified edges (after degree-2 node removal)
        self.NPA_shape = None  # Base NPA polygons

        # Buffer-specific parameters, updated in the process_buffers loop
        self.buffer_mile = None
        self.buffer_distance_feet = None
        self.buffer_mile_name = None

        self._load_base_data()

    def _load_base_data(self):
        """Loads and performs initial cleaning on the global datasets."""

        # 1. Load and clean global edges: remove retired streets
        self.source_edges = gpd.read_file(self.source_edges_path)

        # 2. Load NPA polygons: these define the study area boundaries
        self.NPA_shape = gpd.read_file(self.NPA_shape_path)

        print("Base data loaded and initialized.")
        
    def _validate_crs(self):
        if self.source_edges.crs != self.NPA_shape.crs:
            raise ValueError(
                f"CRS mismatch:\nEdges: {self.source_edges.crs}\nNPA: {self.NPA_shape.crs}\n"
                "Please reproject before running."
            )
    # --------------------------------------------------------------------------
    ## Step 1: Node Extraction (Mapping street endpoints to nodes)
    # --------------------------------------------------------------------------
    def extract_nodes(self):
        """
        Identifies unique endpoints for every street segment, creates a node GeoDataFrame,
        and updates the edge GeoDataFrame with corresponding 'start_node' and 'end_node' IDs.
        """
        print("Starting node extraction (Step 1)...")
        endpoint_coords = []
        node_id = 0
        node_id_mapping = {}  # Key: (x, y) coordinate tuple, Value: unique integer node ID
        crs = self.source_edges.crs

        total_edges = len(self.source_edges)  # Get total count for tqdm

        # TQDM INTEGRATION: Wrap the iteration over edges
        for index, row in tqdm(self.source_edges.iterrows(), total=total_edges, desc="Extracting Nodes"):
            geometry = row['geometry']
            start_point = Point(geometry.coords[0])
            end_point = Point(geometry.coords[-1])

            # Process Start Point
            if start_point.coords[0] not in node_id_mapping:
                endpoint_coords.append({'id': node_id, 'geometry': start_point})
                node_id_mapping[start_point.coords[0]] = node_id
                start_node_id = node_id
                node_id += 1
            else:
                start_node_id = node_id_mapping[start_point.coords[0]]

            # Process End Point
            if end_point.coords[0] not in node_id_mapping:
                endpoint_coords.append({'id': node_id, 'geometry': end_point})
                node_id_mapping[end_point.coords[0]] = node_id
                end_node_id = node_id
                node_id += 1
            else:
                end_node_id = node_id_mapping[end_point.coords[0]]

            self.source_edges.at[index, 'start_node'] = int(start_node_id)
            self.source_edges.at[index, 'end_node'] = int(end_node_id)

        self.source_nodes = gpd.GeoDataFrame(endpoint_coords, crs=crs)
        self.source_nodes['osmid'] = self.source_nodes["id"].astype("int")
        self.source_nodes.set_index('id', inplace=True)
        print(f"Nodes extracted successfully. Total unique nodes: {len(self.source_nodes)}")

    # --------------------------------------------------------------------------
    ## Step 2: Simplify Network (Removing degree-2 nodes)
    # --------------------------------------------------------------------------
    def simplify_network(self):
        """
        Simplifies the global street network by merging sequential edges that are
        connected by a node with a degree of exactly two.
        """
        print("Starting network simplification (Step 2)...")

        edges_df = self.source_edges.copy()
        nodes_df = self.source_nodes.copy()

        # Calculate node degrees
        nodes_start = edges_df['start_node'].value_counts()
        nodes_end = edges_df['end_node'].value_counts()
        node_counts = nodes_start.add(nodes_end, fill_value=0)
        nodes_with_two_edges = node_counts[node_counts == 2].index.tolist()  # Get list of nodes to process

        def merge_edges_for_node(node, edges_df):
            """Core geometry merging logic for a single degree-2 node."""

            connected_edges = edges_df[(edges_df['start_node'] == node) | (edges_df['end_node'] == node)]

            if len(connected_edges) != 2: return edges_df

            geoms = connected_edges['geometry'].tolist()
            common_point_geom = nodes_df[nodes_df.index == node].geometry.iloc[0]
            common_point = (common_point_geom.x, common_point_geom.y)

            coords1 = list(geoms[0].coords)
            coords2 = list(geoms[1].coords)

            # Determine the two new endpoints (the non-common nodes)
            edge1, edge2 = connected_edges.iloc[0], connected_edges.iloc[1]

            e1_other_node = edge1['end_node'] if edge1['start_node'] == node else edge1['start_node']
            e2_other_node = edge2['end_node'] if edge2['start_node'] == node else edge2['start_node']

            new_start_node = e1_other_node
            new_end_node = e2_other_node

            # Create the new LineString geometry (Handling coordinate ordering)

            # Align coords1: it must end at the common point
            if Point(coords1[-1]).coords[0] != common_point: coords1.reverse()

            # Align coords2: it must start at the common point
            if Point(coords2[0]).coords[0] != common_point: coords2.reverse()

            new_coords = coords1[:-1] + coords2
            new_geom = LineString(new_coords)

            # Update the GeoDataFrame
            edges_df = edges_df.drop(connected_edges.index)  # Remove old edges

            new_edge = pd.DataFrame({
                'start_node': new_start_node,
                'end_node': new_end_node,
                'geometry': [new_geom]
            })
            return pd.concat([edges_df, new_edge], ignore_index=True)

        # TQDM INTEGRATION: Apply merging function with a progress bar
        print(f"Total degree-2 nodes to process: {len(nodes_with_two_edges)}")
        for node in tqdm(nodes_with_two_edges, desc="Simplifying Network (Merging Edges)"):
            edges_df = merge_edges_for_node(node, edges_df)

        self.edges_version_2 = gpd.GeoDataFrame(edges_df, geometry='geometry', crs=self.source_edges.crs)
        print(f"Network simplification complete. Remaining edges: {len(self.edges_version_2)}")

    # --------------------------------------------------------------------------
    ## Main Control Flow
    # --------------------------------------------------------------------------
    def process_buffers(self, buffer_list_mile):
        """
        Controls the analysis flow by iterating through a list of buffer distances.
        """
        results = {}

        for self.buffer_mile in buffer_list_mile:
            self.buffer_distance_feet = self.buffer_mile * 5280  # Convert miles to feet
            self.buffer_mile_name = str(self.buffer_mile).replace(".", "").replace("0",
                                                                                   "0") if self.buffer_mile < 1 and self.buffer_mile != 0.1 else str(
                int(self.buffer_mile * 10))

            print(f"\n--- Processing Buffer: {self.buffer_mile} miles ({self.buffer_mile_name}) ---")

            npa_gdf_results = self._process_single_buffer()
            results[self.buffer_mile_name] = npa_gdf_results

        self._merge_results(results)

    def _process_single_buffer(self):
        """
        Applies buffering, extracts local networks, and calculates measures
        for all NPAs at the current buffer distance, showing progress with tqdm.
        """

        # 1. Prepare NPA polygons: buffer them
        buffered_NPA = self.NPA_shape.copy()
        buffered_NPA['geometry'] = buffered_NPA['geometry'].apply(
            lambda geom: geom.buffer(self.buffer_distance_feet)
        )

        results_list = []

        # 🚨 TQDM INTEGRATION: Wrap the iteration over NPA rows
        buffer_desc = f"Analyzing NPAs ({self.buffer_mile} mi)"

        for idx, polygon in tqdm(buffered_NPA.iterrows(), total=len(buffered_NPA), desc=buffer_desc):
            NPA_id = polygon['id']

            # --- Step 3: Extract Local Network ---
            edge_NPA = self.edges_version_2[self.edges_version_2.intersects(polygon.geometry)].copy()

            if edge_NPA.empty:
                measures = {f"{m}{self.buffer_mile_name}": np.nan for m in
                            ["int_den", "st_den", "nd_deg", "cl_cof", "shtpth", "bt_cnt", "cl_cnt", "pr_cnt"]}
                results_list.append({'id': NPA_id, **measures})
                continue

            nodes_NPA, edge_NPA = self._extract_local_elements(edge_NPA)

            # --- Step 4: Extract Largest Component ---
            G_local = self._create_networkx_graph(nodes_NPA, edge_NPA)
            G_main, nodes_main, edges_main = self._extract_largest_component(G_local, nodes_NPA, edge_NPA)

            # --- Step 5: Calculate Network Measures ---
            measures = self._calculate_network_measures(G_main, polygon, nodes_main, edges_main)
            results_list.append({'id': NPA_id, **measures})

        # Save buffer-specific results
        npa_gdf = self.NPA_shape[['id', 'geometry']].merge(
            pd.DataFrame(results_list), on='id', how='left'
        )
        print(f"Results for buffer {self.buffer_mile} generated successfully!")

        return npa_gdf.drop(columns=['geometry'])

        # --------------------------------------------------------------------------

    ## Step 3 Helpers: Local Network Element Extraction
    # --------------------------------------------------------------------------

    def _extract_local_elements(self, edge_NPA):
        """
        Creates a new, locally-indexed set of nodes for the given set of local edges.
        """
        edge_NPA['geometry'] = edge_NPA['geometry'].apply(self._convert_multilinestring_to_linestring)

        endpoint_coords = []
        node_id = 0
        node_id_mapping = {}

        for index, row in edge_NPA.iterrows():
            orig_start_id = row['start_node']
            orig_end_id = row['end_node']

            geometry = row['geometry']
            start_point = Point(geometry.coords[0])
            end_point = Point(geometry.coords[-1])

            if orig_start_id not in node_id_mapping:
                endpoint_coords.append({'id': node_id, 'geometry': start_point, 'orig_id': orig_start_id})
                node_id_mapping[orig_start_id] = node_id
                start_node_id = node_id
                node_id += 1
            else:
                start_node_id = node_id_mapping[orig_start_id]

            if orig_end_id not in node_id_mapping:
                endpoint_coords.append({'id': node_id, 'geometry': end_point, 'orig_id': orig_end_id})
                node_id_mapping[orig_end_id] = node_id
                end_node_id = node_id
                node_id += 1
            else:
                end_node_id = node_id_mapping[orig_end_id]

            edge_NPA.at[index, 'start_node'] = int(start_node_id)
            edge_NPA.at[index, 'end_node'] = int(end_node_id)

        nodes_NPA = gpd.GeoDataFrame(endpoint_coords, crs=edge_NPA.crs)
        nodes_NPA['osmid'] = nodes_NPA["id"].astype("int")
        nodes_NPA.set_index('id', inplace=True)
        return nodes_NPA, edge_NPA

    def _convert_multilinestring_to_linestring(self, geom):
        """Converts a MultiLineString to a single LineString by concatenating coordinates."""
        if isinstance(geom, MultiLineString):
            coords = []
            for part in geom.geoms:
                coords.extend(part.coords)
            return LineString(coords)
        return geom

    # --------------------------------------------------------------------------
    ## Step 4 Helpers: Graph Creation and Largest Component Extraction
    # --------------------------------------------------------------------------

    def _create_networkx_graph(self, nodes_gdf, edges_gdf):
        """Creates a NetworkX graph from the local nodes and edges GeoDataFrames."""
        G = nx.Graph()

        for _, row in nodes_gdf.iterrows():
            G.add_node(row['osmid'], **row.to_dict())

        for _, row in edges_gdf.iterrows():
            G.add_edge(
                int(row['start_node']),
                int(row['end_node']),
                geometry=row['geometry'],
                weight=row['geometry'].length
            )
        return G

    def _extract_largest_component(self, G_local, nodes_NPA, edge_NPA):
        """Identifies and extracts the largest fully connected component of the local network."""
        components = list(nx.connected_components(G_local))

        if not components:
            return nx.Graph(), nodes_NPA.head(0), edge_NPA.head(0)

        largest_component = max(components, key=len)
        G_main = G_local.subgraph(largest_component).copy().to_undirected()

        main_nodes = list(G_main.nodes())
        nodes_main = nodes_NPA[nodes_NPA['osmid'].isin(main_nodes)].copy()
        edges_main = edge_NPA[
            edge_NPA['start_node'].isin(main_nodes) & edge_NPA['end_node'].isin(main_nodes)
            ].copy()

        return G_main, nodes_main, edges_main

    # --------------------------------------------------------------------------
    ## Step 5 Helpers: Network Measure Calculation
    # --------------------------------------------------------------------------

    def _calculate_network_measures(self, G, polygon, nodes_main, edges_main):
        """
        Calculates a full suite of graph-theoretic measures for the final,
        largest-component local network.
        """

        if G.number_of_nodes() < 2:
            return {f"{m}{self.buffer_mile_name}": np.nan for m in
                    ["int_den", "st_den", "nd_deg", "cl_cof", "shtpth", "bt_cnt", "cl_cnt", "pr_cnt"]}

        measures = {}

        # --- Density Measures ---
        area_acre = polygon["geometry"].area / 43560

        total_st_length = edges_main['geometry'].length.sum()
        measures["st_den"] = total_st_length / area_acre

        intersections = [node for node in G.nodes if G.degree(node) >= 3]
        measures["int_den"] = len(intersections) / area_acre

        # --- Local Structure Measures ---
        measures["nd_deg"] = sum(dict(G.degree()).values()) / G.number_of_nodes()
        measures["cl_cof"] = nx.average_clustering(G)

        # --- Centrality & Efficiency Measures ---
        measures["cl_cnt"] = np.mean(list(nx.closeness_centrality(G).values()))
        measures["pr_cnt"] = np.mean(list(nx.pagerank(G).values()))
        measures["bt_cnt"] = np.mean(list(nx.betweenness_centrality(G, weight='weight').values()))

        # Average Shortest Path
        path_lengths = []
        for source in G.nodes():
            try:
                lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
                for target, length in lengths.items():
                    if source != target:
                        path_lengths.append(length)
            except nx.NetworkXNoPath:
                continue

        avg_shortest_path_length = np.mean(path_lengths) if path_lengths else np.nan
        measures["shtpth"] = (avg_shortest_path_length / 5280) if not np.isnan(avg_shortest_path_length) else np.nan

        # Format column names
        return {f"{k}{self.buffer_mile_name}": v for k, v in measures.items()}

    # --------------------------------------------------------------------------
    ## Final Step: Merge Results
    # --------------------------------------------------------------------------

    def _merge_results(self, results):
        """
        Merges the results from all processed buffer sizes into one final
        Analytical Base Table (ABT) shapefile.
        """

        npa_merged = self.NPA_shape.copy()

        for _, df in results.items():
            npa_merged = npa_merged.merge(df, on='id', how='left')

        final_path = os.path.join(os.path.dirname(self.NPA_shape_path), "Final_dataset.shp")
        npa_merged.to_file(final_path)
        print("\nFinal analytical base table (Final_dataset.shp) created successfully!")
