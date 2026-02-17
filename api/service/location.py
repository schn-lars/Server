from shapely.geometry import Point
from pathlib import Path

swiss_shp = Path("../resources/map/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp")
gdf = None
cantons_gdf = { 'Genève': 'GE', 'Thurgau': 'TG', 'Valais': 'VS', 'Aargau': 'AG', 'Schwyz': 'SZ', 'Zürich': 'ZH', 'Obwalden': 'OW',
                'Fribourg': 'FR', 'Glarus': 'GL', 'Uri' : 'UR', 'Nidwalden' : 'NW', 'Solothurn' : 'SO', 'Appenzell Ausserrhoden' : 'AR',
                'Jura' : 'JU', 'Graubünden' : 'GR', 'Vaud' : 'VD', 'Luzern' : 'LU', 'Ticino' : 'TI', 'Zug' : 'ZG', 'Basel-Landschaft' : 'BL',
                'St. Gallen' : 'SG', 'Schaffhausen' : 'SH', 'Bern' : 'BE', 'Basel-Stadt' : 'BS', 'Neuchâtel': 'NE', 'Appenzell Innerrhoden': 'AI'
                }

def get_canton(lat, lon):
    global gdf
    point = Point(lon, lat)
    match = gdf[gdf.contains(point)]
    if not match.empty:
        return cantons_gdf[match.iloc[0]['NAME']]
    return None