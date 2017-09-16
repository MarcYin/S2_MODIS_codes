import ogr
import shapely.geometry
import shapely.wkt
from fastRWpkl import *

def get_polygons(shapefile="./shapefile/wrs2_descending.shp"):
    shapefile = ogr.Open(shapefile)
    # Get the only layer within it
    layer = shapefile.GetLayer(0)

    polygons = []

    # For each feature in the layer
    for i in range(layer.GetFeatureCount()):
        # Get the feature, and its path and row attributes
        feature = layer.GetFeature(i)
        path = feature['PATH']
        row = feature['ROW']

        # Get the geometry into a Shapely-compatible
        # format by converting to Well-known Text (Wkt)
        # and importing that into shapely
        geom = feature.GetGeometryRef()
        shape = shapely.wkt.loads(geom.ExportToWkt())

        # Store the shape and the path/row values
        # in a list so we can search it easily later
        polygons.append((shape, path, row))
    return polygons

def get_wrs( lat, lon):
    """Get the Landsat WRS-2 path and row for the given
    latitude and longitude co-ordinates.

    Returns a list of dicts, as some points will be in the
    overlap between two (or more) landsat scene areas:

    [{path: 202, row: 26}, {path:186, row=7}]
    """

    # Create a point with the given latitude
    # and longitude (NB: the arguments are lon, lat
    # not lat, lon)
    pt = shapely.geometry.Point(lon, lat)
    res = []
    if glob.glob('pkls/polygons*') != []:
        polygons = parallel_rw_pkl(None, 'polygons', 'r')
    # Iterate through every polgon
    else:
        polygons = get_polygons()
    for poly in polygons:
        # If the point is within the polygon then
        # append the current path/row to the results
        # list
        if pt.within(poly[0]):
            res.append({'path': poly[1], 'row': poly[2], 'poly': poly[0]})

    # Return the results list to the user
    return res