-- ============================================================
--  Urban Street Network & Property Crime — PostGIS schema
--  Charlotte / Mecklenburg County, 371 NPAs
--  CRS: EPSG:2264 (NC State Plane, feet)
-- ============================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- ── 1. Core NPA layer ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS npa (
    npa_id      TEXT PRIMARY KEY,
    area_sqft   DOUBLE PRECISION,   -- original CRS units (feet²)
    area_acre   DOUBLE PRECISION,
    geom        GEOMETRY(GEOMETRY, 2264)
);

CREATE INDEX IF NOT EXISTS idx_npa_geom ON npa USING GIST (geom);

-- ── 2. Crime outcomes ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS npa_crime (
    npa_id          TEXT PRIMARY KEY REFERENCES npa (npa_id),
    propcrim_ct     INTEGER,            -- raw incident count
    propcrim_1k     DOUBLE PRECISION,   -- per 1,000 residents
    propcrim_acre   DOUBLE PRECISION    -- per acre (density)
);

-- ── 3. Socioeconomic & housing controls ───────────────────
CREATE TABLE IF NOT EXISTS npa_socioeconomic (
    npa_id              TEXT PRIMARY KEY REFERENCES npa (npa_id),
    bach_deg            DOUBLE PRECISION,   -- % bachelor's degree or higher
    employment          DOUBLE PRECISION,   -- % employed
    home_ownership      DOUBLE PRECISION,   -- % owner-occupied
    res_occupancy       DOUBLE PRECISION,   -- % residential occupancy
    singlefam_rate      DOUBLE PRECISION,   -- % single-family units
    singlefam_age       DOUBLE PRECISION,   -- median age of single-family stock
    housing_density     DOUBLE PRECISION,   -- units per acre
    singlefam_size      DOUBLE PRECISION,   -- median heated sq ft
    median_sale_price   DOUBLE PRECISION,   -- median sale price ($)
    median_age          DOUBLE PRECISION,   -- median resident age
    median_income       DOUBLE PRECISION,   -- median household income ($)
    race_other          DOUBLE PRECISION,   -- % other race
    race_asian          DOUBLE PRECISION,
    race_black          DOUBLE PRECISION,
    race_latin          DOUBLE PRECISION,
    race_white          DOUBLE PRECISION,
    heated_area         DOUBLE PRECISION    -- total heated sq ft in NPA
);

-- ── 4. Street-network measures (tidy / long format) ───────
--
--  metric values  (8 measures × 5 buffer distances = 40 rows per NPA)
--
--  buffer_mi ∈ {0.1, 0.25, 0.5, 0.75, 1.0}
--  metric    ∈ {int_den, st_den, nd_deg, cl_cof,
--               shtpth,  bt_cnt, cl_cnt, pr_cnt}
--
CREATE TABLE IF NOT EXISTS npa_network_measures (
    npa_id      TEXT    NOT NULL REFERENCES npa (npa_id),
    buffer_mi   NUMERIC(4,2) NOT NULL,
    metric      TEXT    NOT NULL,
    value       DOUBLE PRECISION,
    PRIMARY KEY (npa_id, buffer_mi, metric)
);

CREATE INDEX IF NOT EXISTS idx_nnm_metric    ON npa_network_measures (metric);
CREATE INDEX IF NOT EXISTS idx_nnm_buffer    ON npa_network_measures (buffer_mi);
CREATE INDEX IF NOT EXISTS idx_nnm_npa       ON npa_network_measures (npa_id);

-- ── 5. Convenience view: wide feature matrix ──────────────
--  Joins all tables into one row per NPA — mirrors the shape
--  of final_dataset/extracted_measures.shp
CREATE OR REPLACE VIEW v_feature_matrix AS
SELECT
    n.npa_id,
    n.area_acre,
    n.geom,
    -- crime
    c.propcrim_ct,
    c.propcrim_1k,
    c.propcrim_acre,
    -- socioeconomic
    s.bach_deg, s.employment, s.home_ownership, s.res_occupancy,
    s.singlefam_rate, s.singlefam_age, s.housing_density, s.singlefam_size,
    s.median_sale_price, s.median_age, s.median_income,
    s.race_other, s.race_asian, s.race_black, s.race_latin, s.race_white,
    s.heated_area,
    -- network measures (pivoted)
    MAX(CASE WHEN m.metric='int_den' AND m.buffer_mi=0.10 THEN m.value END) AS int_den_01,
    MAX(CASE WHEN m.metric='int_den' AND m.buffer_mi=0.25 THEN m.value END) AS int_den_025,
    MAX(CASE WHEN m.metric='int_den' AND m.buffer_mi=0.50 THEN m.value END) AS int_den_05,
    MAX(CASE WHEN m.metric='int_den' AND m.buffer_mi=0.75 THEN m.value END) AS int_den_075,
    MAX(CASE WHEN m.metric='int_den' AND m.buffer_mi=1.00 THEN m.value END) AS int_den_10,
    MAX(CASE WHEN m.metric='st_den'  AND m.buffer_mi=0.10 THEN m.value END) AS st_den_01,
    MAX(CASE WHEN m.metric='st_den'  AND m.buffer_mi=0.25 THEN m.value END) AS st_den_025,
    MAX(CASE WHEN m.metric='st_den'  AND m.buffer_mi=0.50 THEN m.value END) AS st_den_05,
    MAX(CASE WHEN m.metric='st_den'  AND m.buffer_mi=0.75 THEN m.value END) AS st_den_075,
    MAX(CASE WHEN m.metric='st_den'  AND m.buffer_mi=1.00 THEN m.value END) AS st_den_10,
    MAX(CASE WHEN m.metric='nd_deg'  AND m.buffer_mi=0.10 THEN m.value END) AS nd_deg_01,
    MAX(CASE WHEN m.metric='nd_deg'  AND m.buffer_mi=0.25 THEN m.value END) AS nd_deg_025,
    MAX(CASE WHEN m.metric='nd_deg'  AND m.buffer_mi=0.50 THEN m.value END) AS nd_deg_05,
    MAX(CASE WHEN m.metric='nd_deg'  AND m.buffer_mi=0.75 THEN m.value END) AS nd_deg_075,
    MAX(CASE WHEN m.metric='nd_deg'  AND m.buffer_mi=1.00 THEN m.value END) AS nd_deg_10,
    MAX(CASE WHEN m.metric='cl_cof'  AND m.buffer_mi=0.10 THEN m.value END) AS cl_cof_01,
    MAX(CASE WHEN m.metric='cl_cof'  AND m.buffer_mi=0.25 THEN m.value END) AS cl_cof_025,
    MAX(CASE WHEN m.metric='cl_cof'  AND m.buffer_mi=0.50 THEN m.value END) AS cl_cof_05,
    MAX(CASE WHEN m.metric='cl_cof'  AND m.buffer_mi=0.75 THEN m.value END) AS cl_cof_075,
    MAX(CASE WHEN m.metric='cl_cof'  AND m.buffer_mi=1.00 THEN m.value END) AS cl_cof_10,
    MAX(CASE WHEN m.metric='shtpth'  AND m.buffer_mi=0.10 THEN m.value END) AS shtpth_01,
    MAX(CASE WHEN m.metric='shtpth'  AND m.buffer_mi=0.25 THEN m.value END) AS shtpth_025,
    MAX(CASE WHEN m.metric='shtpth'  AND m.buffer_mi=0.50 THEN m.value END) AS shtpth_05,
    MAX(CASE WHEN m.metric='shtpth'  AND m.buffer_mi=0.75 THEN m.value END) AS shtpth_075,
    MAX(CASE WHEN m.metric='shtpth'  AND m.buffer_mi=1.00 THEN m.value END) AS shtpth_10,
    MAX(CASE WHEN m.metric='bt_cnt'  AND m.buffer_mi=0.10 THEN m.value END) AS bt_cnt_01,
    MAX(CASE WHEN m.metric='bt_cnt'  AND m.buffer_mi=0.25 THEN m.value END) AS bt_cnt_025,
    MAX(CASE WHEN m.metric='bt_cnt'  AND m.buffer_mi=0.50 THEN m.value END) AS bt_cnt_05,
    MAX(CASE WHEN m.metric='bt_cnt'  AND m.buffer_mi=0.75 THEN m.value END) AS bt_cnt_075,
    MAX(CASE WHEN m.metric='bt_cnt'  AND m.buffer_mi=1.00 THEN m.value END) AS bt_cnt_10,
    MAX(CASE WHEN m.metric='cl_cnt'  AND m.buffer_mi=0.10 THEN m.value END) AS cl_cnt_01,
    MAX(CASE WHEN m.metric='cl_cnt'  AND m.buffer_mi=0.25 THEN m.value END) AS cl_cnt_025,
    MAX(CASE WHEN m.metric='cl_cnt'  AND m.buffer_mi=0.50 THEN m.value END) AS cl_cnt_05,
    MAX(CASE WHEN m.metric='cl_cnt'  AND m.buffer_mi=0.75 THEN m.value END) AS cl_cnt_075,
    MAX(CASE WHEN m.metric='cl_cnt'  AND m.buffer_mi=1.00 THEN m.value END) AS cl_cnt_10,
    MAX(CASE WHEN m.metric='pr_cnt'  AND m.buffer_mi=0.10 THEN m.value END) AS pr_cnt_01,
    MAX(CASE WHEN m.metric='pr_cnt'  AND m.buffer_mi=0.25 THEN m.value END) AS pr_cnt_025,
    MAX(CASE WHEN m.metric='pr_cnt'  AND m.buffer_mi=0.50 THEN m.value END) AS pr_cnt_05,
    MAX(CASE WHEN m.metric='pr_cnt'  AND m.buffer_mi=0.75 THEN m.value END) AS pr_cnt_075,
    MAX(CASE WHEN m.metric='pr_cnt'  AND m.buffer_mi=1.00 THEN m.value END) AS pr_cnt_10
FROM npa n
JOIN npa_crime        c ON c.npa_id = n.npa_id
JOIN npa_socioeconomic s ON s.npa_id = n.npa_id
JOIN npa_network_measures m ON m.npa_id = n.npa_id
GROUP BY n.npa_id, n.area_acre, n.geom,
         c.propcrim_ct, c.propcrim_1k, c.propcrim_acre,
         s.bach_deg, s.employment, s.home_ownership, s.res_occupancy,
         s.singlefam_rate, s.singlefam_age, s.housing_density, s.singlefam_size,
         s.median_sale_price, s.median_age, s.median_income,
         s.race_other, s.race_asian, s.race_black, s.race_latin, s.race_white,
         s.heated_area;
