#!/usr/bin/env python
"""
Usage: coordinates.py

Contains basic functions to manipulate points.
"""
from typing import List
import shapefile

def get_lower_long_coordinate(shapes: List[shapefile.Shape]):
    '''Get the lower longitude from list of (long, lat) bounding boxes
        Parameters:
            shapes (List[shapefile.Shape]): list of shapes
        Returns:
            lower_long (str): west most longitude
    '''
    lower_long = shapes[0].bbox[0]
    for shape in shapes[1:]:
        if shape.bbox[0] < lower_long:
            lower_long = shape.bbox[0]
    return lower_long

def get_greater_long_coordinate(shapes: List[shapefile.Shape]):
    '''Get the greater longitude from list of (long, lat) bounding box
        Parameters:
            shapes (List[shapefile.Shape]): list of shapes
        Returns:
            greater_long (str): east most longitude
    '''
    greater_long = shapes[0].bbox[2]
    for shape in shapes[1:]:
        if shape.bbox[2] > greater_long:
            greater_long = shape.bbox[2]
    return greater_long

def get_lower_lat_coordinate(shapes: List[shapefile.Shape]):
    '''Get the lower latitude from list of (long, lat) bounding boxes
        Parameters:
            shapes (List[shapefile.Shape]): list of shapes
        Returns:
            lower_lat (str): south most longitude
    '''
    lower_lat = shapes[1:][0].bbox[1]
    for shape in shapes:
        if shape.bbox[1] < lower_lat:
            lower_lat = shape.bbox[1]
    return lower_lat

def get_greater_lat_coordinate(shapes: List[shapefile.Shape]):
    '''Get the greater latitude from list of (long, lat) bounding boxes
        Parameters:
            shapes (List[shapefile.Shape]): list of shapes
        Returns:
            greater_lat (str): north most longitude
    '''
    greater_lat = shapes[0].bbox[3]
    for shape in shapes[1:]:
        if shape.bbox[3] > greater_lat:
            greater_lat = shape.bbox[3]
    return greater_lat

def get_bounding_box_zone(shapes: List[shapefile.Shape]):
    '''Get the long,lat bbox coordinates of the bbox of the whole zone represented by shapes
        Parameters:
            shapes (List[shapefile.Shape]): list of shapes
        Returns:
            zone_shap (List): list of (long, lat) tuples for the bounding box of the zone from East/North in clockwise order
    '''
    lower_long = get_lower_long_coordinate(shapes)
    greater_long = get_greater_long_coordinate(shapes)
    lower_lat = get_lower_lat_coordinate(shapes)
    greater_lat = get_greater_lat_coordinate(shapes)
    zone_shape = ((lower_long, greater_lat), (greater_long, greater_lat), (greater_long, lower_lat), (lower_long, lower_lat))
    return zone_shape