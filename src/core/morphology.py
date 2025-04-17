#!/usr/bin/env python
"""
Usage: display.py

Contains advanced functions apply morphological ...
"""
from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn
import shapefile
from src.utils.display import draw_cc_from_contours, draw_cc_best_enclosing_rectangle

FACTOR = 10 # multiplicative factor on meters distance to get better precision

def get_cv2_contours_from_shapes(shapes: List[shapefile.Shape]):
    '''Get contours from shapes
        Parameters:
            shapes (List[shapefile.Shape]): list of shapes
        Returns:
            contours (List[numpy.ndarray]): List of int32 numpy arrays contours in the form (n, 1, 2) where n is the number of points in polygon
    '''
    contours = []
    for shape in shapes:
        contour_uint32 = (np.array(shape.points)*FACTOR).astype(np.int32) 
        contours.append(contour_uint32[:,np.newaxis,:])
    return contours

    
def find_cc_best_enclosing_rectangles(shapes: List[shapefile.Shape], contours: List[Any], index_display: int = -1):
    '''Find the best enclosing rectangle for each shape (height, width, angle, center of the best rectangle)
        Parameters:
            shapes (List[shapefile.Shape]): list of shapes
            contours (List[numpy.ndarray]): List of int32 numpy arrays contours in the form (n, 1, 2) where n is the number of points in polygon
            index_display (int): index of shape for which we will display the best enclosing rectangle (for debugging purpose)
    '''
    # draw contours
    image_mask = draw_cc_from_contours(contours, False)
    # loop over polygon contour to get the best enclosing rectange using opencv
    for index, (shape, contour) in enumerate(zip(shapes, contours)):
        best_rectangle  = cv2.minAreaRect(contour)
        ((center_x, center_y), (width_, height_), angle_) = best_rectangle
        best_rectangle_points = cv2.boxPoints(best_rectangle)
        best_rectangle_points = best_rectangle_points[:,np.newaxis,:].astype(np.int32)
        ytop = np.min(best_rectangle_points[:, :, 1])
        ybottom = np.max(best_rectangle_points[:, :, 1])
        xleft = np.min(best_rectangle_points[:, :, 0])
        xright = np.max(best_rectangle_points[:, :, 0])
        image_mask_cropped = image_mask[ytop:ybottom, xleft:xright]
        # get number of white pixels in the connected component
        area = (image_mask_cropped[:, :, 0]>0).sum()
        print(area)
        # height should always be the longest dimension
        if height_ < width_:
            height_, width_ = width_, height_
        shape.rectangle_center_x = center_x/FACTOR # remove the multiplicative factor to get the true meters
        shape.rectangle_center_y = center_y/FACTOR
        shape.rectangle_width = width_/FACTOR
        shape.rectangle_height = height_/FACTOR
        shape.rectangle_angle = angle_
        shape.vehicles_area = area/(FACTOR*FACTOR)
        if index == index_display:
            # draw enclosed rectangle for a specific shape index and crop
            draw_cc_best_enclosing_rectangle(image_mask, best_rectangle)

def find_cc_outliers(shapes_normalized: List[shapefile.Shape], contours: List[Any], thresh: int = -3, plot: bool = True):
    '''Divide shapes into 3 groups, the two last are sub categories of outliers:
            - single wehicle not well segmented or falseala
            - many vehicles merged
        Parameters:
            shapes (List[shapefile.Shape]): list of shapes
            contours (List[numpy.ndarray]): List of int32 numpy arrays contours in the form (n, 1, 2) where n is the number of points in polygon
            thresh (float): threshold under which we consider a the segment is an outlier among two subcategories
                - single not well segmented vehicle or fa
                - merged vehicle
            plot (bool): plot colored connected component to identify the two outliers groups
        Returns:
            contours_category1 (List[Any]): category of well segmented single vehicles
            contours_category2 (List[Any]): subcategory of outliers containing not well segmented wehicle or fa
            contours_category3 (List[Any]): subcategory of outliers containing merged vehicles
            index_category1 (List[int]): indexes of category of well segmented single vehicles
            index_category2 (List[int]): indexes of subcategory of outliers containing not well segmented wehicle or fa
            index_category3 (List[int]): indexex of subcategory of outliers containing merged vehicles
            mean_size (float): mean size vehicle from fitted gaussian
    '''
    # get best height, width rectangle for each connected component
    hv = [shape.rectangle_height for shape in shapes_normalized]
    wv = [shape.rectangle_width for shape in shapes_normalized]
    # apply a Gaussian Mixture with one component to the (hv, wv) distribution
    gm = sklearn.mixture.GaussianMixture(n_components=1) # we use a mixture of gaussians model with only one component
    X = np.concatenate((np.array(hv)[:,np.newaxis],np.array(wv)[:,np.newaxis]), axis=1)
    gm.fit(X)
    # compute log likelihood of each component
    scorev = gm.score_samples(X)
    # loop over connected component and enrich the 3 groups
    contours_category1 = [] # vehicle well segmented
    contours_category2 = [] # single vehicle not well segmented or false alarm
    contours_category3 = [] # many vehicles merged
    index_category1 = [] # vehicle well segmented
    index_category2 = [] # single vehicle not well segmented or false alarm
    index_category3 = [] # many vehicles merged
    mean_size = gm.means_[0][0] * gm.means_[0][1]
    for cc_index, (cc_h, cc_w, cc_score, contour) in enumerate(zip(hv, wv, scorev, contours)):
        if cc_score > thresh:
            contours_category1.append(contour)
            index_category1.append(cc_index)
        elif cc_h*cc_w < mean_size:
            contours_category2.append(contour)
            index_category2.append(cc_index)
        else:
            contours_category3.append(contour)
            index_category3.append(cc_index)

    return mean_size, contours_category1, contours_category2, contours_category3, index_category1, index_category2, index_category3
    