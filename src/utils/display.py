#!/usr/bin/env python
"""
Usage: display.py

Contains basic functions to display geometrics, shapes, etc...
"""
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2

arg: int = 0
def draw_cc_from_contours(contours: List[np.ndarray], plot: bool =True, color: tuple = (255,255,255), height: int = -1, width: int = -1):
    '''Draw shapes from mask from polygon contours
        Parameters:
            height (int): desired height for image plot (will be infered if None)
            width (int): desired width for image plot (will be infered if None)
            contours (List[numpy.ndarray]): List of int32 numpy arrays contours in the form (n, 1, 2) where n is the number of points in polygon
            plot (bool): whether to show image using matplotlib
        Returns:
            image_mask (np.ndarray): image containing white filled polygon in black background
    '''

    # get width, height image
    if width < 0:
        width = np.max(np.concatenate(contours, axis=0)[:,:,0]) + 100 # add margin
    if height < 0:
        height = np.max(np.concatenate(contours, axis=0)[:,:,1]) + 100
    # draw contours
    image_mask = np.zeros((height, width)).astype(np.uint8)
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_mask, tuple(contours), -1, color, cv2.FILLED)
    if plot:
        plt.figure(figsize=(14,14))
        # origin is at top, left for image, so flip image horizontally
        image_mask = np.flipud(image_mask)
        # plot image
        plt.imshow(image_mask)
        plt.title("mask segmentation reconstructed using polygon in meters")
        plt.show()
    return image_mask


def draw_cc_best_enclosing_rectangle(image_mask: np.ndarray, best_rectangle: tuple):
    '''Draw the best enclosing rectangle for a shape using associated cv2 contour
        Parameters
            image_mask (np.ndarray): image containing white filled polygon in black background
            best_rectangle (tuple): rotated rectangle in the form ((center_x, center_y), (width_, height_), angle_)
        Returns:
            image_best_rectangle (np.ndarray): image with best rectangle inside enclosing the target contour
    '''
    # draw rectangle
    best_rectangle_points = cv2.boxPoints(best_rectangle)
    best_rectangle_points = best_rectangle_points[:,np.newaxis,:].astype(np.int32)
    image_mask = cv2.polylines(image_mask, [best_rectangle_points], isClosed=True, color=(255,0,0), thickness = 2)
    # crop image
    ytop = np.min(best_rectangle_points[:, :, 1])
    ybottom = np.max(best_rectangle_points[:, :, 1])
    xleft = np.min(best_rectangle_points[:, :, 0])
    xright = np.max(best_rectangle_points[:, :, 0])
    image_mask_cropped = image_mask[ytop:ybottom, xleft:xright]
    image_mask_cropped = np.flipud(image_mask_cropped)
    plt.imshow(image_mask_cropped)
    plt.show()

def draw_cc_categories_from_contours(contours_good_category: List[np.ndarray], contours_outliers_subcategory1: List[np.ndarray], contours_outliers_subcategory2: List[np.ndarray]):
    '''Draw categories contours
        Parameters
            contours(List[np.ndarray]): all contours
            contours_good_category (List[np.ndarray]): category of well segmented single vehicles
            contours_outliers_subcategory1 (List[np.ndarray]): subcategory of outliers containing not well segmented wehicle or fa
            contours_outliers_subcategory2 (List[np.ndarray]): subcategory of outliers containing merged vehicles
    '''
    # Define width, height image
    all_contours = contours_good_category + contours_outliers_subcategory1 + contours_outliers_subcategory1
    width = np.max(np.concatenate(all_contours, axis=0)[:,:,0]) + 100 # add margin
    height = np.max(np.concatenate(all_contours, axis=0)[:,:,1]) + 100

    # first subcategory in yellow (not well segmented single vehicle or fa)
    image_mask1 = draw_cc_from_contours(contours_outliers_subcategory1, color=(255, 255, 0), plot=False, height=height, width=width)
    # second category in red (merged vehicles)
    image_mask2 = draw_cc_from_contours(contours_outliers_subcategory2, color=(255, 0, 0), plot=False, height=height, width=width)
    # good contours
    image_mask3 = draw_cc_from_contours(contours_good_category, color=(255, 255, 255), plot=False, height=height, width=width)
    
    image_mask = image_mask1 + image_mask2 + image_mask3
    plt.figure(figsize=(12,12))
    # origin is at top, left for image, so flip image horizontally
    image_mask = np.flipud(image_mask)
    # plot image
    plt.imshow(image_mask)
    plt.title("mask segmentation with conncted components divided into outliers subcategories")
    plt.show()