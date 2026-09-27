import pandas as pd
import json
import ast
import numpy as np

cols = ['year', 'damage_property_usd', 'deaths_direct', 'injuries_direct', 'area_name', 'state_fips', 'county_fips', 'location_points']
df = pd.read_csv("data/processed_data/flood_1996_2026.csv", usecols=cols)


def extract_coords(val):
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


df['coords'] = df['location_points'].apply(extract_coords)
df = df.dropna(subset=['coords'])

df['damage'] = df['damage_property_usd'].fillna(0)
df['deaths'] = df['deaths_direct'].fillna(0)

time_series_data = {}
for yr, sub_df in df.groupby('year'):
    events = []
    for _, row in sub_df.iterrows():
        events.append({"c": row['coords'], "d": round(float(row['damage'])), "f": int(row['deaths']), "n": str(row['area_name'])})
    time_series_data[int(yr)] = events

with open("flood_timeseries_1996_2026.json", "w") as f:
    json.dump(time_series_data, f)
