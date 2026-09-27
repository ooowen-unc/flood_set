import os
import pandas as pd
import json
import ast
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

cols = ['year', 'begin_time_local', 'event_type', 'damage_property_usd', 'deaths_direct', 'area_name', 'state_fips', 'county_fips', 'location_points']
df = pd.read_csv(os.path.join(BASE, "data", "processed_data", "flood_1996_2026.csv"), usecols=cols)

df = df[(df['year'] >= 1999) & (df['year'] <= 2025)]

cpi = pd.read_csv(os.path.join(BASE, "CPIAUCSL.csv"), parse_dates=['observation_date'])
cpi['month_key'] = cpi['observation_date'].dt.strftime('%Y-%m')
cpi_sorted = cpi.sort_values('observation_date')
cpi_by_month = dict(zip(cpi_sorted['month_key'], cpi_sorted['CPIAUCSL']))
CPI_TARGET = cpi_sorted.iloc[-1]['CPIAUCSL']
cpi_keys_sorted = sorted(cpi_by_month.keys())


def cpi_factor(month_key):
    # nearest CPI month at or before the event month
    idx = np.searchsorted(cpi_keys_sorted, month_key, side='right') - 1
    if idx < 0:
        return 1.0
    base = cpi_by_month[cpi_keys_sorted[idx]]
    return CPI_TARGET / base if base > 0 else 1.0


def parse_coords(val):
    if pd.isna(val):
        return None
    try:
        pts = ast.literal_eval(val)
        if pts and len(pts) > 0:
            lat, lon = pts[0][0], pts[0][1]
            if 20 <= lat <= 55 and -130 <= lon <= -65:
                return [round(lon, 3), round(lat, 3)]
    except:
        pass
    return None


df['coords'] = df['location_points'].apply(parse_coords)

fips_coord_map = df.dropna(subset=['coords', 'county_fips']).drop_duplicates(subset=['county_fips']).set_index('county_fips')['coords'].to_dict()
df['coords'] = df.apply(lambda r: fips_coord_map.get(r['county_fips']) if r['coords'] is None else r['coords'], axis=1)
df = df.dropna(subset=['coords'])

df['month'] = pd.to_datetime(df['begin_time_local'], errors='coerce').dt.month.fillna(1).astype(int)
df['damage'] = df['damage_property_usd'].fillna(0)
df['deaths'] = df['deaths_direct'].fillna(0)
df = df[(df['damage'] > 0) | (df['deaths'] > 0)]

df['is_flash'] = df['event_type'].astype(str).str.contains('Flash', case=False).astype(int)

df['lon_bin'] = df['coords'].apply(lambda c: round(c[0] * 2) / 2)
df['lat_bin'] = df['coords'].apply(lambda c: round(c[1] * 2) / 2)

time_series_data = {}
running_total_loss = 0

for yr in range(1999, 2026):
    for m in range(1, 13):
        key = f"{yr}-{m:02d}"
        sub_df = df[(df['year'] == yr) & (df['month'] == m)]
        if sub_df.empty:
            continue

        grouped = sub_df.groupby(['lon_bin', 'lat_bin', 'is_flash']).agg({'damage': 'sum', 'deaths': 'sum', 'area_name': 'first', 'coords': 'first'}).reset_index()

        factor = cpi_factor(key)

        events = []
        month_loss = 0
        month_loss_cpi = 0
        for _, r in grouped.iterrows():
            d_val = round(float(r['damage']))
            dc_val = round(d_val * factor)
            month_loss += d_val
            month_loss_cpi += dc_val
            events.append({"c": r['coords'], "d": d_val, "dc": dc_val, "f": int(r['deaths']), "n": str(r['area_name']), "t": int(r['is_flash'])})

        running_total_loss += month_loss
        time_series_data[key] = {"events": events, "month_loss": month_loss, "month_loss_cpi": month_loss_cpi, "cum_loss": running_total_loss}

with open(os.path.join(BASE, "flood_timeseries_aggregated.json"), "w") as f:
    json.dump(time_series_data, f)
