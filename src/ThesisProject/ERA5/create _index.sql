CREATE INDEX idx_era5_calc_ts_id ON era5_calculated_data (ts_id);
CREATE INDEX idx_era5_calc_latitude ON era5_calculated_data (latitude);
CREATE INDEX idx_era5_calc_longitude ON era5_calculated_data (longitude);
CREATE INDEX idx_era5_calc_lat_lon ON era5_calculated_data (latitude, longitude);
CREATE INDEX idx_era5_calc_lat_lon_ts_id ON era5_calculated_data (latitude, longitude, ts_id);

CREATE INDEX idx_era5_ts_id ON era5_raw_data (ts_id);
CREATE INDEX idx_era5_latitude ON era5_raw_data (latitude);
CREATE INDEX idx_era5_longitude ON era5_raw_data (longitude);
CREATE INDEX idx_era5_lat_lon ON era5_raw_data (latitude, longitude);
CREATE INDEX idx_era5_lat_lon_ts_id ON era5_raw_data (latitude, longitude, ts_id);

CREATE INDEX idx_prod_ts_id ON productions (ts_id);
CREATE INDEX idx_prod_wf_id ON productions (wf_id);
CREATE INDEX idx_prod_wf_id_ts_id ON productions (wf_id, ts_id);