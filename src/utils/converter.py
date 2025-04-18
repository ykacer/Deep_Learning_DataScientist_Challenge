#!/usr/bin/env python
"""
Usage: converter.py

Contains basic functions to project points.
"""
from typing import List
import shapefile
import utm
from src.utils.coordinates import get_lower_long_coordinate, get_lower_lat_coordinate

def project_and_normalize_shapes(shapes: List[shapefile.Shape]):
    '''Create new shapes with long,lat projected in meters (using UTM convention) and normalized using 
            - south-most, west-most bounding box point as origin
            - x-axis from west to east
            - y-axis from south to north
        Parameters:
            shapes (List[shapefile.Shape]): list of shapes
        Returns:
            shapes_normalized (List[shapefile.Shape]): list of shapes where bbox and polygon are in meters and normalized
    '''
    # get the minimal (x,y) to substract later
    lower_long = get_lower_long_coordinate(shapes)
    lower_lat = get_lower_lat_coordinate(shapes)
    (lower_x, lower_y) = utm.from_latlon(lower_lat, lower_long)[:2]
    
    # loop over shapes to convert to meters and substract minimal (x,y)
    shapes_normalized_list = []
    for shape_index, shape in enumerate(shapes):
        normalized_bbox = [0, 0, 0, 0]
        # project lon,lat in utm system to get x,y meters coordinates
        normalized_bbox[0], normalized_bbox[1] = utm.from_latlon(shape.bbox[1], shape.bbox[0])[:2]# get (x,y) from (lat, lont)
        normalized_bbox[2], normalized_bbox[3] = utm.from_latlon(shape.bbox[3], shape.bbox[2])[:2]
        # normalize the x,y coordinates of bbox by substracting min(x), min(y)
        normalized_bbox[0] = normalized_bbox[0] - lower_x
        normalized_bbox[2] = normalized_bbox[2] - lower_x
        normalized_bbox[1] = normalized_bbox[1] - lower_y
        normalized_bbox[3] = normalized_bbox[3] - lower_y
        normalized_points = []
        for point in shape.points:
            # project polygon point lon,lat in utm system to get x,y meters coordinates
            normalized_point = list(utm.from_latlon(point[1], point[0])[:2])
            # normalize the x,y coordinates of polygon point by substracting min(x), min(y)
            normalized_point[0] = normalized_point[0] - lower_x
            normalized_point[1] = normalized_point[1] - lower_y
            normalized_points.append(normalized_point)
        # instantiate normalized shape
        shape_normalized = shapefile.Shape(shapeType=shape.shapeType,
                                           points=normalized_points,
                                           oid=shape.oid)
        shape_normalized.bbox = bbox=normalized_bbox
        # enrich the list of normalized shapes
        shapes_normalized_list.append(shape_normalized)

    # instantiate the normalized Shapes from the list
    shapes_normalized = shapefile.Shapes(shapes_normalized_list)
    return shapes_normalized
