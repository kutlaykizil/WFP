

-- WORK IN PROGRESS


/* load spatialite  */
/* WELL APPARENTLY I NEED TO RUN THE BELOW QUERY EVERY TIME */
SELECT load_extension('/usr/lib/x86_64-linux-gnu/mod_spatialite.so');
SELECT InitSpatialMetaData();


/* GEOMETRY */

/* Only run the followings when there is a new wind farm entry */

/* Create point geometry for wt from lat-long. */
-- Add the geometry column if it doesn't exist

-- drop the geom column if it exists
ALTER TABLE wf_turbine_coordinates drop column geom;
SELECT AddGeometryColumn('wf_turbine_coordinates', 'geom', 3035, 'POINT', 'XY');
UPDATE wf_turbine_coordinates
SET geom = ST_Transform(
                ST_GeomFromText(
                    'POINT(' || ux || ' ' || uy || ')',
                    (CASE
                        WHEN um = 27 THEN 23035
                        WHEN um = 33 THEN 23036
                        WHEN um = 39 THEN 23037
                        WHEN um = 45 THEN 23038
                    END)
                ),
                3035
           );

/* Create point geometry for wf_border_coordinates from lat-long. */

-- drop the geom column
ALTER TABLE wf_border_coordinates drop column geom;
-- Add geometry column
SELECT AddGeometryColumn('wf_border_coordinates', 'geom', 3035, 'POINT', 'XY');
UPDATE wf_border_coordinates
SET geom = ST_Transform(
                    ST_GeomFromText('POINT(' || kx || ' ' || ky || ')',
                       (CASE
                           WHEN km = 27 THEN 23035
                           WHEN km = 33 THEN 23036
                           WHEN km = 39 THEN 23037
                           WHEN km = 45 THEN 23038
                       END)
                    ),
                    3035
                );








---------------------------------------------------------------------
-- CREATE POLYGON GEOMETRIES IN 'wf' TABLE
---------------------------------------------------------------------

-- Step 1: Add a new GEOMETRY column to the 'wf' table for the farm boundary polygons.
-- Using 'farm_boundary_geom' to distinguish from the old TEXT 'geom_polygon'.
SELECT DiscardGeometryColumn('wf', 'farm_boundary_geom');
SELECT AddGeometryColumn('wf', 'farm_boundary_geom', 3035, 'POLYGON', 'XY');

-- Step 2: Construct and update polygons using WKT created with GROUP_CONCAT.
-- This method builds the 'POLYGON((x1 y1, x2 y2, ..., x1 y1))' WKT string.
WITH OrderedPointCoords AS (
    -- Selects coordinates and the coordinates of the first point for closing the ring
    SELECT
        wf_id,
        point_index,
        ST_X(geom) as x,
        ST_Y(geom) as y,
        FIRST_VALUE(ST_X(geom)) OVER (PARTITION BY wf_id ORDER BY point_index ASC) as first_x,
        FIRST_VALUE(ST_Y(geom)) OVER (PARTITION BY wf_id ORDER BY point_index ASC) as first_y
    FROM wf_border_coordinates
    WHERE geom IS NOT NULL -- Only use valid points
),
PolygonWKT AS (
    -- Creates the WKT string for a POLYGON's exterior ring
    SELECT
        wf_id,
        -- Ensure there are enough distinct points to form a polygon
        -- The string must start with POLYGON(( and end with ))
        -- The coordinate pairs are x y, separated by commas
        -- The last point must be the same as the first to close the ring
        'POLYGON((' || GROUP_CONCAT(x || ' ' || y, ',' ORDER BY point_index ASC) || ',' || first_x || ' ' || first_y || '))' as wkt_string
    FROM OrderedPointCoords
    GROUP BY wf_id
    HAVING COUNT(DISTINCT point_index) >= 3 -- Need at least 3 distinct points for a valid ring
)
-- Update the 'wf' table with polygons created from the WKT strings
UPDATE wf
SET farm_boundary_geom = (
    SELECT ST_GeomFromText(pw.wkt_string, 3035)
    FROM PolygonWKT pw
    WHERE pw.wf_id = wf.wf_id AND pw.wkt_string IS NOT NULL
)
WHERE EXISTS (
    SELECT 1
    FROM PolygonWKT pw
    WHERE pw.wf_id = wf.wf_id AND pw.wkt_string IS NOT NULL
);

-- Optional: You can then update your original 'geom_polygon' TEXT column if needed,
-- or drop it and rename 'farm_boundary_geom'.
-- Example: UPDATE wf SET geom_polygon = ST_AsText(farm_boundary_geom) WHERE farm_boundary_geom IS NOT NULL;

-- You might also want to create a spatial index on the new geometry column for performance:
SELECT CreateSpatialIndex('wf', 'farm_boundary_geom');


