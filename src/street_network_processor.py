import os, io, csv, json, time, hashlib, warnings, numpy as np, pandas as pd, geopandas as gpd, networkx as nx, scipy
from shapely.geometry import Point, LineString, MultiLineString
from tqdm import tqdm
warnings.filterwarnings("ignore")

# The eight configuration measures, in output order (see README for definitions)
NETWORK_METRICS = ["int_den", "st_den", "nd_deg", "cl_cof", "shtpth", "bt_cnt", "cl_cnt", "pr_cnt"]

FINAL_DIR = "../data/final_dataset"
CHECKPOINT_DIR = "../data/checkpoints"   # one sub-folder per pair of input layers, one CSV per buffer
SAME_CRS_TOLERANCE = 0.1                 # layer units (feet): two CRS definitions that move no coordinate
                                         # by more than this are the same system written differently


def crs_label(crs):
    """Short readable name of a CRS, e.g. 'EPSG:2264 (NAD83 / North Carolina (ftUS))'."""
    if crs is None:
        return "none"
    code = crs.to_epsg()
    name = crs.name or crs.to_string()
    return f"EPSG:{code} ({name})" if code else name


def align_neighborhoods_crs(edges, neighborhoods):
    """
    Put the neighborhoods in the CRS of the streets layer.

    The streets define the working CRS (all lengths are measured on them); the neighborhoods
    are the small layer, so they are the one reprojected when the two differ. Two definitions
    of the same system (an EPSG code vs. the WKT written by desktop GIS software, say) are
    recognised by reprojecting and checking that no coordinate moved: then the geometry is
    kept as it is and only the CRS definition of the streets is adopted.
    Returns (neighborhoods, note); note is None when nothing had to change.
    """
    if edges.crs is None or neighborhoods.crs is None or edges.crs == neighborhoods.crs:
        return neighborhoods, None
    moved = neighborhoods.to_crs(edges.crs)
    shift = float(np.nanmax(np.abs(moved.bounds.to_numpy() - neighborhoods.bounds.to_numpy())))
    if shift < SAME_CRS_TOLERANCE:
        note = (f"the neighborhoods CRS is the streets CRS written differently "
                f"(no coordinate moves by more than {shift:.3g} ft); kept as is")
        return neighborhoods.set_crs(edges.crs, allow_override=True), note
    note = (f"neighborhoods reprojected from {crs_label(neighborhoods.crs)} "
            f"to the streets CRS {crs_label(edges.crs)}")
    return moved, note


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

        # 1. Load global edges (assumed already cleaned by the user)
        self.source_edges = gpd.read_file(self.source_edges_path)

        # 2. Load NPA polygons: these define the study area boundaries
        self.NPA_shape = gpd.read_file(self.NPA_shape_path)
        if "id" not in self.NPA_shape.columns:
            raise ValueError("Neighborhood layer must have a unique identifier column named 'id'.")

        # 3. Edges and polygons must share a projected CRS (buffers are in feet);
        #    the neighborhoods are reprojected to the streets CRS when they differ
        self._validate_crs()

        print(f"Base data loaded: {len(self.source_edges)} edges, {len(self.NPA_shape)} neighborhoods.")
        
    def _validate_crs(self):
        for name, gdf in (("Street", self.source_edges), ("Neighborhood", self.NPA_shape)):
            if gdf.crs is None:
                raise ValueError(f"{name} layer has no coordinate reference system defined.")
        self.NPA_shape, note = align_neighborhoods_crs(self.source_edges, self.NPA_shape)
        if note:
            print(f"Note: {note}")

    def _as_lines(self, geom):
        """Return a list of LineStrings for a geometry (LineString/MultiLineString)."""
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, LineString):
            return [geom]
        if isinstance(geom, MultiLineString):
            return list(geom.geoms)
        return []

    def _to_linestring(self, geom):
        """Ensure geometry is a LineString (flatten MultiLineString if needed)."""
        if isinstance(geom, MultiLineString):
            coords = []
            for part in geom.geoms:
                coords.extend(list(part.coords))
            return LineString(coords)
        return geom

    # --------------------------------------------------------------------------
    ## Step 1: Node Extraction (Mapping street endpoints to nodes)
    # --------------------------------------------------------------------------
    def extract_nodes(self):
        print("Starting node extraction (Step 1)...")
        endpoint_coords = []
        node_id = 0
        node_id_mapping = {}  # Key: (x, y), Value: node id
        crs = self.source_edges.crs

        total_edges = len(self.source_edges)

        for index, row in tqdm(self.source_edges.iterrows(), total=total_edges, desc="Extracting Nodes"):
            geom = row.geometry

            lines = self._as_lines(geom)
            if not lines:
                continue

            # Use the first and last part to define endpoints for multipart features
            first_line = lines[0]
            last_line = lines[-1]

            start_xy = tuple(first_line.coords[0])
            end_xy = tuple(last_line.coords[-1])

            # Start node
            if start_xy not in node_id_mapping:
                endpoint_coords.append({'id': node_id, 'geometry': Point(start_xy)})
                node_id_mapping[start_xy] = node_id
                start_node_id = node_id
                node_id += 1
            else:
                start_node_id = node_id_mapping[start_xy]

            # End node
            if end_xy not in node_id_mapping:
                endpoint_coords.append({'id': node_id, 'geometry': Point(end_xy)})
                node_id_mapping[end_xy] = node_id
                end_node_id = node_id
                node_id += 1
            else:
                end_node_id = node_id_mapping[end_xy]

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

            geoms = [self._to_linestring(g) for g in connected_edges['geometry'].tolist()]
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
    @staticmethod
    def buffer_code(buffer_mile):
        """Suffix used in output column names: 0.1 -> '1', 0.25 -> '025', 0.5 -> '05',
        0.75 -> '075', 1.0 -> '10' (kept short so shapefile field names stay <= 10 chars)."""
        if buffer_mile < 1 and buffer_mile != 0.1:
            return str(buffer_mile).replace(".", "")
        return str(int(buffer_mile * 10))

    @classmethod
    def measure_columns(cls, buffer_list_mile):
        """Names of the measure columns written for these buffers, in output order."""
        return [f"{m}{cls.buffer_code(b)}" for b in buffer_list_mile for m in NETWORK_METRICS]

    def process_buffers(self, buffer_list_mile, recompute=False):
        """
        Measures every buffer in turn, then assembles the output table.

        The measures are checkpointed as they are produced: one CSV per buffer in
        data/checkpoints/<input fingerprint>/, to which each neighborhood's row is appended as
        soon as it has been measured. A run that stops (error, Ctrl+C, power) therefore resumes
        where it was, and a buffer already complete for the same inputs is not computed again.
        recompute=True ignores the existing checkpoints of the buffers in this run.
        """
        checkpoint_dir = self._checkpoint_dir()
        print(f"Checkpoints: {checkpoint_dir}")

        for self.buffer_mile in buffer_list_mile:
            self.buffer_distance_feet = self.buffer_mile * 5280  # Convert miles to feet
            self.buffer_mile_name = self.buffer_code(self.buffer_mile)

            print(f"\n--- Processing Buffer: {self.buffer_mile} miles ({self.buffer_mile_name}) ---")

            self._process_single_buffer(self._checkpoint_path(checkpoint_dir, self.buffer_mile), recompute)

        self._merge_results(buffer_list_mile, checkpoint_dir)

    # --------------------------------------------------------------------------
    ## Checkpoints
    # --------------------------------------------------------------------------

    def _input_fingerprint(self):
        """
        Short hash of what the network measures depend on: the street geometries and the
        neighborhood ids and geometries (in the working CRS). Attribute edits, e.g. a corrected
        control variable, leave it unchanged, so they do not invalidate the checkpoints; other
        street or neighborhood geometries get a checkpoint folder of their own.
        """
        h = hashlib.sha1()
        h.update(b"".join(w for w in self.source_edges.geometry.to_wkb() if w is not None))
        h.update("|".join(self.NPA_shape["id"].astype(str)).encode("utf-8"))
        h.update(b"".join(w for w in self.NPA_shape.geometry.to_wkb() if w is not None))
        return h.hexdigest()[:10]

    def _checkpoint_dir(self):
        """Checkpoint folder for the current inputs, with an inputs.json saying what they were."""
        path = os.path.join(CHECKPOINT_DIR, self._input_fingerprint())
        os.makedirs(path, exist_ok=True)
        info = os.path.join(path, "inputs.json")
        if not os.path.exists(info):
            with open(info, "w", encoding="utf-8") as f:
                json.dump({
                    "streets": self.source_edges_path,
                    "neighborhoods": self.NPA_shape_path,
                    "segments": int(len(self.source_edges)),
                    "neighborhood_count": int(len(self.NPA_shape)),
                    "crs": crs_label(self.source_edges.crs),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "note": "Street-network measures computed from these inputs: one CSV per buffer, one "
                            "row per neighborhood, written as each neighborhood is measured. The folder "
                            "name is a fingerprint of the street and neighborhood geometries, so other "
                            "inputs get their own folder. Delete the folder, or run with --recompute, "
                            "to measure from scratch.",
                }, f, indent=2)
        return path

    @classmethod
    def _checkpoint_path(cls, checkpoint_dir, buffer_mile):
        return os.path.join(checkpoint_dir, f"measures_{cls.buffer_code(buffer_mile)}.csv")

    @staticmethod
    def _read_checkpoint(path, columns):
        """
        Rows already measured for one buffer (an empty table when there is no checkpoint).
        A last line left incomplete by an interruption (no newline) is dropped.
        """
        if not os.path.exists(path):
            return pd.DataFrame(columns=columns)
        with open(path, "rb") as f:
            raw = f.read()
        if raw and not raw.endswith(b"\n"):
            raw = raw[:raw.rfind(b"\n") + 1]
        if not raw.strip():
            return pd.DataFrame(columns=columns)
        df = pd.read_csv(io.BytesIO(raw), dtype={"id": str})
        if list(df.columns) != columns:
            raise ValueError(f"{path} has columns {list(df.columns)}, expected {columns}; "
                             "delete the file or run with --recompute")
        return df.drop_duplicates("id", keep="last")

    def _process_single_buffer(self, checkpoint_path, recompute=False):
        """
        Buffers the neighborhoods, extracts each local network and measures it at the current
        buffer distance, appending every neighborhood's row to the checkpoint as soon as it is
        done; neighborhoods already in the checkpoint are skipped (unless recompute).
        """
        columns = ["id"] + [f"{m}{self.buffer_mile_name}" for m in NETWORK_METRICS]
        done = pd.DataFrame(columns=columns) if recompute else self._read_checkpoint(checkpoint_path, columns)
        all_ids = self.NPA_shape["id"].astype(str)
        todo = self.NPA_shape[~all_ids.isin(set(done["id"]))]

        if len(done):
            print(f"Checkpoint: {len(done)} of {len(all_ids)} neighborhoods already measured, "
                  f"{len(todo)} to go.")
        if todo.empty:
            print(f"Results for buffer {self.buffer_mile} taken from the checkpoint.")
            return

        # 1. Prepare NPA polygons: buffer the ones still to do
        buffered_NPA = todo.copy()
        buffered_NPA['geometry'] = buffered_NPA['geometry'].buffer(self.buffer_distance_feet)

        # 2. Rewrite the checkpoint cleanly (header + rows already done), then append as we go
        with open(checkpoint_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(columns)
            for row in done[columns].itertuples(index=False):
                writer.writerow(["" if (isinstance(v, float) and np.isnan(v)) else v for v in row])
            f.flush()

            buffer_desc = f"Analyzing NPAs ({self.buffer_mile} mi)"
            for idx, polygon in tqdm(buffered_NPA.iterrows(), total=len(buffered_NPA), desc=buffer_desc):
                NPA_id = polygon['id']

                # --- Step 3: Extract Local Network ---
                edge_NPA = self.edges_version_2[self.edges_version_2.intersects(polygon.geometry)].copy()

                if edge_NPA.empty:
                    measures = {f"{m}{self.buffer_mile_name}": np.nan for m in NETWORK_METRICS}
                else:
                    nodes_NPA, edge_NPA = self._extract_local_elements(edge_NPA)

                    # --- Step 4: Extract Largest Component ---
                    G_local = self._create_networkx_graph(nodes_NPA, edge_NPA)
                    G_main, nodes_main, edges_main = self._extract_largest_component(G_local, nodes_NPA, edge_NPA)

                    # --- Step 5: Calculate Network Measures ---
                    measures = self._calculate_network_measures(G_main, polygon, nodes_main, edges_main)

                # --- Checkpoint: the row is on disk before the next neighborhood starts ---
                values = [measures[c] for c in columns[1:]]
                writer.writerow([str(NPA_id)] + ["" if (isinstance(v, float) and np.isnan(v)) else v
                                                 for v in values])
                f.flush()
                os.fsync(f.fileno())

        print(f"Results for buffer {self.buffer_mile} generated successfully! "
              f"({len(todo)} neighborhoods measured now, {len(done)} from the checkpoint)")

        # --------------------------------------------------------------------------

    ## Step 3 Helpers: Local Network Element Extraction
    # --------------------------------------------------------------------------

    def _extract_local_elements(self, edge_NPA):
        """
        Creates a new, locally-indexed set of nodes for the given set of local edges.
        """
        edge_NPA['geometry'] = edge_NPA['geometry'].apply(self._to_linestring)

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
            return {f"{m}{self.buffer_mile_name}": np.nan for m in NETWORK_METRICS}

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

    def _merge_results(self, buffer_list_mile, checkpoint_dir):
        """
        Assembles the final Analytical Base Table (ABT) shapefile from the checkpoints of
        the requested buffers: the neighborhoods layer with the measure columns appended.
        """
        npa_merged = self.NPA_shape.copy()
        key = npa_merged["id"].astype(str)

        for buffer_mile in buffer_list_mile:
            code = self.buffer_code(buffer_mile)
            columns = ["id"] + [f"{m}{code}" for m in NETWORK_METRICS]
            df = self._read_checkpoint(self._checkpoint_path(checkpoint_dir, buffer_mile), columns)
            missing = set(key) - set(df["id"])
            if missing:
                raise RuntimeError(f"buffer {buffer_mile} mi: {len(missing)} neighborhoods are not in the "
                                   "checkpoint; run again to complete it")
            df = df.set_index("id").reindex(key)
            for c in columns[1:]:
                npa_merged[c] = df[c].to_numpy(dtype=float)

        os.makedirs(FINAL_DIR, exist_ok=True)
        final_path = os.path.join(FINAL_DIR, "extracted_measures.shp")
        npa_merged.to_file(final_path)
        print(f"\nFinal analytical base table written to {final_path}")
