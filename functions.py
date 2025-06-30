import configparser
import math
import os
import time

import monai
import numpy as np
import scipy as sp
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from skimage.measure import label
from sympy import Eq, Symbol, cos, sin
from sympy.solvers import solve
from aux import center_scan, visualize_sitk_image_3d, validate_landmarks, trim_skull_as_sphere, trim_outlier_region
import pyvista


def align(nii_path, output_path, debug=False):
    """ Aligns a CT head by landmarks of cochleas and nasal bridge, similar to the orbito-meatal line.

    Parameters:
        nii_path (str): Full path to a CT head in .nii format
        output_path (str): Full path including .nii filename to output the aligned CT head
    """
    input_config = configparser.ConfigParser()
    # Get the directory of functions.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for model files in models directory    
    models_dir = os.path.join(current_dir, "models")
    if not os.path.exists(models_dir):
        raise ValueError(f"Models directory not found at {models_dir}")

    print("Using the landmark models from:", models_dir)
            
    # Find model files containing nb, rc, lc
    model_files = os.listdir(models_dir)
    nb_model_path = ""
    rc_model_path = ""
    lc_model_path = ""
    
    for model_file in model_files:
        if "nb" in model_file.lower():
            nb_model_path = os.path.join(models_dir, model_file)
        elif "rc" in model_file.lower():
            rc_model_path = os.path.join(models_dir, model_file)
        elif "lc" in model_file.lower():
            lc_model_path = os.path.join(models_dir, model_file)
            
    if not (nb_model_path and rc_model_path and lc_model_path):
        raise ValueError("Could not find all required model files (nb, rc, lc) in models directory")
    
    pitch_degrees = None
    roll_degrees = None
    yaw_degrees = None
    transforms = None
    
    input_sitk = nii2sitk(nii_path)
    print("Predicting the landmarks...")

    max_iterations = 3
    iteration = 0

    spacing=(1.5,1.5,1.5)
    
    while iteration < max_iterations:
        
        check = False #used to determine the correctness of landmarks

        #if debug:
        #    visualize_sitk_image_3d(input_sitk, threshold=0)

        orig_slice_thickness = input_sitk.GetSpacing()[2]
        if orig_slice_thickness > 1.0:
            raise ValueError('Cannot process. Sent slice thickness was ' + str(orig_slice_thickness) + ' mm - needs to be < 1.0 mm')
        if orig_slice_thickness <= 1.0:
            aligned_sitk = input_sitk

            while not check:
                nb_label_image = predict_low_res_interpolate(input_sitk=aligned_sitk, model_path=nb_model_path, spacing=spacing, roi=(96,96,96), num_labels=1, verbose=False)
                rc_label_image = predict_low_res_interpolate(input_sitk=aligned_sitk, model_path=rc_model_path, spacing=spacing, roi=(96,96,96), num_labels=1, verbose=False)
                lc_label_image = predict_low_res_interpolate(input_sitk=aligned_sitk, model_path=lc_model_path, spacing=spacing, roi=(96,96,96), num_labels=1, verbose=False)

                check, nb_center, rc_center, lc_center  = validate_landmarks(nb_label_image, rc_label_image, lc_label_image)
                
                if(nb_center == None): 
                    if(spacing != 1): 
                        spacing = tuple(np.array(spacing) - 0.1)
                        print("Reducing spacing to " + str(spacing) + " mm")
                        continue

                    else:
                        print("Spacing: 1mm, No nasal bridge was found. ID: " + os.path.basename(nii_path))
                        return # If no nasal bridge was found, return

                if not check:
                    #aligned_sitk = trim_outlier_region(aligned_sitk, nb_center, rc_center, lc_center)

                    #keep only the skull region
                    aligned_sitk = trim_skull_as_sphere(aligned_sitk, nb_center, rc_center, lc_center)

            if debug:
                #combine three of the label images into one            
                output_path_no_filename = os.path.split(output_path)[0]
                sitk.WriteImage(input_sitk, os.path.join(output_path_no_filename, 'orig_ct.nii.gz'))
                sitk.WriteImage(nb_label_image, os.path.join(output_path_no_filename, 'nb_label_image.nii.gz'))
                sitk.WriteImage(rc_label_image, os.path.join(output_path_no_filename, 'rc_label_image.nii.gz'))
                sitk.WriteImage(lc_label_image, os.path.join(output_path_no_filename, 'lc_label_image.nii.gz'))            

            # Determine the location of the segmentations in mm
            nb_label_image_data = sitk.GetArrayFromImage(nb_label_image)
            nb_label_image_data = np.transpose(nb_label_image_data, (2, 1, 0))
            nb_index = sp.ndimage.measurements.center_of_mass(nb_label_image_data)
            nb_index = (int(nb_index[0]), int(nb_index[1]), int(nb_index[2]))
            nb_mm = tuple(np.multiply(nb_index, input_sitk.GetSpacing()))

            rc_label_image_data = sitk.GetArrayFromImage(rc_label_image)
            rc_label_image_data = np.transpose(rc_label_image_data, (2, 1, 0))
            rc_index = sp.ndimage.measurements.center_of_mass(rc_label_image_data)
            rc_index = (int(rc_index[0]), int(rc_index[1]), int(rc_index[2]))
            rc_mm = tuple(np.multiply(rc_index, input_sitk.GetSpacing()))

            lc_label_image_data = sitk.GetArrayFromImage(lc_label_image)
            lc_label_image_data = np.transpose(lc_label_image_data, (2, 1, 0))
            lc_index = sp.ndimage.measurements.center_of_mass(lc_label_image_data)
            lc_index = (int(lc_index[0]), int(lc_index[1]), int(lc_index[2]))
            lc_mm = tuple(np.multiply(lc_index, input_sitk.GetSpacing()))

            # Calculate the rotation center
            transforms = []
            x, y, z = input_sitk.GetSize()
            rotation_center_index = (int(np.ceil(x/2)), int(np.ceil(y/2)), int(np.ceil(z/2)))
            rotation_center_mm = tuple(np.multiply(rotation_center_index, input_sitk.GetSpacing()))

            # **********************
            # Perform pitch correction
            # **********************
            # Calculate rotation angles - make the nasal bridge and left cochlea align on z axis
            pitch_rads = get_angle_to_match_y(
                (lc_mm[1], lc_mm[2]), 
                (nb_mm[1], nb_mm[2]),
                (rotation_center_mm[1], rotation_center_mm[2])
            )
            # Correction - the pitch rotation is about 15 degrees too steep (too counterclockwise) based on landmarks
            #pitch_rads = pitch_rads + math.radians(15)
            pitch_degrees = math.degrees(pitch_rads)

            if(abs(pitch_degrees) < 3):
                print("Pitch correction is less than 3 degrees, breaking the loop.")
                break # If the pitch correction is less than 3 degrees, break the loop

            # Create transform for rotation
            rotate_transform = get_rotate_transform(input_sitk, angle=pitch_degrees, pitch=True)
            transforms.append(rotate_transform)

            # Calculate the location of landmarks after rotation - rotating around x axis, y and z changing
            nb_mm_yz = get_coords_after_rotation((nb_mm[1], nb_mm[2]), (rotation_center_mm[1], rotation_center_mm[2]), pitch_rads)
            nb_mm = (nb_mm[0], nb_mm_yz[0], nb_mm_yz[1])      
            rc_mm_yz = get_coords_after_rotation((rc_mm[1], rc_mm[2]), (rotation_center_mm[1], rotation_center_mm[2]), pitch_rads)
            rc_mm = (rc_mm[0], rc_mm_yz[0], rc_mm_yz[1])
            lc_mm_yz = get_coords_after_rotation((lc_mm[1], lc_mm[2]), (rotation_center_mm[1], rotation_center_mm[2]), pitch_rads)
            lc_mm = (lc_mm[0], lc_mm_yz[0], lc_mm_yz[1])               

            # **********************
            # Perform yaw correction
            # **********************    
            # Calculate rotation angles - make the cochleas align on the y axis
            yaw_rads = get_angle_to_match_y(
                (lc_mm[0], lc_mm[1]), 
                (rc_mm[0], rc_mm[1]),
                (rotation_center_mm[0], rotation_center_mm[1])
            )
            yaw_degrees = math.degrees(yaw_rads)

            # Create transform for rotation
            rotate_transform = get_rotate_transform(input_sitk, angle=yaw_degrees, yaw=True)
            transforms.append(rotate_transform)
            
            # Calculate the location of landmarks after rotation - rotating around z axis, x and y changing
            nb_mm_xy = get_coords_after_rotation((nb_mm[0], nb_mm[1]), (rotation_center_mm[0], rotation_center_mm[1]), yaw_rads)
            nb_mm = (nb_mm_xy[0], nb_mm_xy[1], nb_mm[2])      
            rc_mm_xy = get_coords_after_rotation((rc_mm[0], rc_mm[1]), (rotation_center_mm[0], rotation_center_mm[1]), yaw_rads)
            rc_mm = (rc_mm_xy[0], rc_mm_xy[1], rc_mm[2])
            lc_mm_xy = get_coords_after_rotation((lc_mm[0], lc_mm[1]), (rotation_center_mm[0], rotation_center_mm[1]), yaw_rads)
            lc_mm = (lc_mm_xy[0], lc_mm_xy[1], lc_mm[2])

            # **********************
            # Perform roll correction
            # **********************

            # Calculate rotation angles - make the cochleas align on the z axis
            roll_rads = get_angle_to_match_y(
                (lc_mm[0], lc_mm[2]), 
                (rc_mm[0], rc_mm[2]),
                (rotation_center_mm[0], rotation_center_mm[2])
            )
            roll_degrees = -math.degrees(roll_rads)
            #roll_degrees = math.degrees(roll_rads)

            # Create transform for rotation
            rotate_transform = get_rotate_transform(input_sitk, angle=roll_degrees, roll=True)
            transforms.append(rotate_transform)

            # Calculate the location of landmarks after rotation - rotating around y axis, x and z changing
            nb_mm_xz = get_coords_after_rotation((nb_mm[0], nb_mm[2]), (rotation_center_mm[0], rotation_center_mm[2]), roll_rads)
            nb_mm = (nb_mm_xz[0], nb_mm[1], nb_mm_xz[1])     
            rc_mm_xz = get_coords_after_rotation((rc_mm[0], rc_mm[2]), (rotation_center_mm[0], rotation_center_mm[2]), roll_rads)
            rc_mm = (rc_mm_xz[0], rc_mm[1], rc_mm_xz[1])
            lc_mm_xz = get_coords_after_rotation((lc_mm[0], lc_mm[2]), (rotation_center_mm[0], rotation_center_mm[2]), roll_rads)
            lc_mm = (lc_mm_xz[0], lc_mm[1], lc_mm_xz[1])

            # **********************
            # Apply transforms
            # **********************
            input_sitk = composite(input_sitk, transforms, interpolation=sitk.sitkBSpline)

            iteration += 1
            print("Iteration " + str(iteration) + " - pitch: " + str(np.around(pitch_degrees, decimals=1)) + ", roll: " + str(np.around(roll_degrees, decimals=1)) + ", yaw: " + str(np.around(yaw_degrees, decimals=1)))

    print("Saving the aligned image to " + output_path)
    sitk.WriteImage(input_sitk, output_path)

def composite(input_image, transforms, interpolation=sitk.sitkBSpline):
    """ Applies a list of transforms as a composite transform
    Parameters:
        input_image (sitk.Image): SimpleITK image, will not be modified
        transforms (list): List of transforms to apply
        interpolation (optional): SimpleITK interpolation method - default is sitk.sitkLinear
    Returns:
        transformed SimpleITK image    
    """
    composite_transform = sitk.CompositeTransform(transforms)

    input_image_data = sitk.GetArrayFromImage(input_image)
    working_image = sitk.GetImageFromArray(input_image_data)
    working_image.CopyInformation(input_image) 

    working_image_data = sitk.GetArrayFromImage(working_image)
    min_value = np.amin(working_image_data)

    # Composite transforms were applied stack-based - first in, last applied
    # With sitk 2.0 there is now a CompositeTransform class. Not sure what order they're applied.
    working_image = sitk.Resample(working_image, composite_transform, interpolation, float(min_value)) # Float required for Resample

    return working_image


def determine_plane(image_sitk):
    """
    Determines the orientation of the study (axial, sagittal, coronal). Must provide either a path to DICOM or a SimpleITK Image

    Parameters:
        input_sitk (SimpleITK.Image, optional): SimpleITK image

    Returns:
        orientation (str): Possible values: axial, sagittal, coronal. None if no input is provided
    """

    orientation = None
    d = image_sitk.GetDirection()

    # Want absolute values of the vectors - maximum movement in any direction
    vector_x = list(map(abs, (d[0], d[3], d[6])))
    vector_y = list(map(abs, (d[1], d[4], d[7])))

    max_x = np.argmax(vector_x)
    max_y = np.argmax(vector_y)

    if max_x == 0 and max_y == 1:
        # Axial: Moving in L-R direction changes x the most, moving in AP direction changes y the most
        orientation = 'axial'
    elif max_x == 1 and max_y == 2:
        # Sagittal: Moving in L-R direction changes y the most, moving in AP direction changes z the most
        orientation = 'sagittal'
    elif max_x == 0 and max_y == 2:
        # Sagittal: Moving in L-R direction changes x the most, moving in AP direction changes z the most
        orientation = 'coronal'

    return orientation


def get_angle_to_match_y(xy_coords1, xy_coords2, rotation_center):
    """ Calculates the rotation angle (in radians) about a point to make the y coordinates of 2 points match
    
    Parameters:
        xy_coords1 (tuple): First x,y coordinates
        xy_coords2 (tuple): Second x,y coordinates
        rotation_center (tuple): Coordinates about which rotation occurs

    Returns:
        angle (radians): Rotation angle necessary to match the y coordinates
    """    
    orig_point1 = xy_coords1
    orig_point2 = xy_coords2

    x1 = orig_point1[0]
    y1 = orig_point1[1]
    x2 = orig_point2[0]
    y2 = orig_point2[1]
    p = rotation_center[0]
    q = rotation_center[1]

    # Determine the rotation to make the y points equal
    # https://keisan.casio.com/exec/system/1223522781
    # https://math.stackexchange.com/questions/270194/how-to-find-the-vertices-angle-after-rotation
    x = Symbol('x')
    #new_y1 = (x1-p)*sin(x) - (y1-q)*cos(x) + q
    #new_y2 = (x2-p)*sin(x) - (y2-q)*cos(x) + q    
    equation = Eq((((x1-p)*sin(x) - (y1-q)*cos(x) + q) - ((x2-p)*sin(x) - (y2-q)*cos(x) + q)), 0) # Solve for the 'x' that makes y1 and y2 equal (y1-y2=0)

    theta = solve(equation, x)
    # Find the smallest absolute angle
    min_theta = theta[np.argmin(np.absolute(theta))]

    return min_theta

def get_coords_after_rotation(orig_coords_mm, rotation_center_mm, angle_rads):
    """ Calculates coordinates after a 2D rotation
    Based on: https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
    
    Parameters:
        orig_coords_mm (list): Coordinates to rotate - e.g. (5.5, 6.2)
        rotation_center_mm (list): Coordinates of center of rotation - e.g. (0,0)
        angle_rads (float): Angle in radians to rotate

    Returns:
        List of coordinates after rotation - e.g. (4.2, 6.3)
    """
    x, y = orig_coords_mm
    ox, oy = rotation_center_mm

    qx = ox + math.cos(angle_rads) * (x - ox) + math.sin(angle_rads) * (y - oy)
    qy = oy + -math.sin(angle_rads) * (x - ox) + math.cos(angle_rads) * (y - oy)
    return qx, qy

def get_rotate_transform(input_image, angle, pitch=False, roll=False, yaw=False, verbose=False):
    """ Generates a transform to rotate an image along 1 axis
    Based on: https://stackoverflow.com/questions/56171643/simpleitk-rotation-of-mri-image

    Parameters:
        input_image: SimpleITK image that has been loaded by the load_sitk method
        angle: Angle in degrees to rotate the image counterclockwise
        pitch (bool): rotate the image in a pitch direction (around x axis)
        roll (bool): rotate the image in a roll direction (around y axis)
        yaw (bool): rotate the image in a yaw direction (around z axis)
        verbose (bool): Verbose flag
    Returns:
        rotated SimpleITK image    
    """
    start_time = None
    if verbose:
        start_time = time.time()
        print("Generating rotate transform..", end = '')
            
    angle_rads = np.deg2rad(angle)
    euler_transform = sitk.Euler3DTransform()
    x, y, z = input_image.GetSize()
    image_center = input_image.TransformIndexToPhysicalPoint((int(np.ceil(x/2)), int(np.ceil(y/2)), int(np.ceil(z/2))))
    euler_transform.SetCenter(image_center)

    direction = input_image.GetDirection()
    axis_angle = None
    
    if pitch:
        axis_angle = (direction[0], direction[3], direction[6], angle_rads)
    elif roll:
        axis_angle = (direction[1], direction[4], direction[7], angle_rads)
    elif yaw:
        axis_angle = (direction[2], direction[5], direction[8], angle_rads)

    if axis_angle is None:
        raise ValueError("get_rotate_transform: pitch, roll, or yaw must be set to True")
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix(np_rot_mat.flatten().tolist())

    if verbose:
        print('.done - time = ' + str(np.around((time.time() - start_time), decimals=1)) + 'sec')
    
    return euler_transform


def keep_only_largest_connected_components(prediction_data):
    """ Keeps the largest component component of each label class (e.g. largest '1', largest '2', etc)

    Parameters:
        prediction_data (numpy Array): Data generated from prediction. Background = 0, labels are integers starting with 1

    Returns:
        prediction_data_new (numpy Array): Data with the largest connected components isolated
    """     
    prediction_data_new = np.zeros_like(prediction_data)
    num_labels = prediction_data.max()
    for label_num in range(1, num_labels + 1): # Iterate through each label number
        prediction_data_tmp = np.zeros_like(prediction_data)
        label_indices = np.where(prediction_data == label_num) # Create a tmp array with only the label number
        prediction_data_tmp[label_indices] = 1
        labels = label(prediction_data_tmp) # Identifies connected components, assigns each component a different number
        if labels.max() != 0:
            largest_cc_indices = labels == np.argmax(np.bincount(labels.flat)[1:])+1 # np.bincount counts the "size" of each component, np.argmax picks the largest one
            prediction_data_new[largest_cc_indices] = label_num
    return prediction_data_new


# This function is from https://github.com/rock-learning/pytransform3d/blob/7589e083a50597a75b12d745ebacaa7cc056cfbd/pytransform3d/rotations.py#L302
def matrix_from_axis_angle(a):
    """ Compute rotation matrix from axis-angle.
    This is called exponential map or Rodrigues' formula.
    https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(theta) +
    #      (1.0 - np.cos(theta)) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(theta))

    return R


def nii2sitk(nii_path, transform_pixels_to_standard_orientation=True, verbose=False):
    """ Loads an nii file as an SITK image

    Parameters:
        nii_path (str): Full path to an .nii or .nii.gz file
        transform_pixels_to_standard_orientation (bool): If true, align pixels to standard LPS orientation

    Returns:
        sitk.Image
    """
    if verbose:
        print('Loading nii as sitk..', end='')
    sitk_image = sitk.ReadImage(nii_path)
    #align image to center
    #sitk_image = center_scan(sitk_image)
    sitk_image_data = sitk.GetArrayFromImage(sitk_image)
    if np.min(sitk_image_data) > -1:
        sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt16)
    else:
        sitk_image = sitk.Cast(sitk_image, sitk.sitkInt16)   

    if transform_pixels_to_standard_orientation:
        sitk_image = pixels_to_standard_orientation(sitk_image)
    if verbose:
        print('.done')
    return sitk_image

def pixels_to_standard_orientation(input_image):
    """ Rotates the pixels of an image to the expected orientation. Important to get expected behavior with rotations.
    In SimpleITK, every image is LPS, meaning the left, posterior, superior-most pixel has the greatest physical location value in mm.
    In standard orientation, the left, posterior, superior-most pixel should have the highest pixel index (x-size-1, y-size-1, z-size-1).
    The right, anterior, inferior-most pixel should be the origin (0,0,0).

    Parameters:
        input_sitk (SimpleITK.Image): Simple ITK image to manipulate

    Returns:
        standard_sitk (SimpleITK.Image): Simple ITK image with pixels in expected orientation
    """
    input_image_data = sitk.GetArrayFromImage(input_image) # z,y,x
    converted_image_data = input_image_data
    
    d = input_image.GetDirection()
    new_x_vector = (d[0], d[3], d[6])
    new_y_vector = (d[1], d[4], d[7])
    new_z_vector = (d[2], d[5], d[8])

    new_origin = None
    new_origin_indices = (0,0,0)

    first_index_location = input_image.TransformIndexToPhysicalPoint((0,0,0))
    last_index_location = input_image.TransformIndexToPhysicalPoint(input_image.GetSize())

    plane = determine_plane(image_sitk=input_image)
    image_modified = False
    if plane == 'axial':
        # Is the greatest x pixel index physically left or right?
        # Greatest physical position (biggest number) is left
        if last_index_location[0] < first_index_location[0]: # The highest x pixel value is right - should be left
            # Flip about the x axis
            converted_image_data = np.flip(converted_image_data, 2)

            # Flip the orientation x component
            new_x_vector = tuple(-np.array(new_x_vector)+0)

            # Flip the origin x component
            new_origin_indices = (input_image.GetSize()[0]-1, new_origin_indices[1], new_origin_indices[2])
            image_modified = True

        # Is the greatest y pixel index physically anterior or posterior?
        # Greatest physical position is posterior
        if last_index_location[1] < first_index_location[1]: # The highest y pixel value is anterior - should be posterior
            # Flip about the y axis
            converted_image_data = np.flip(converted_image_data, 1)

            # Flip the orientation y component
            new_y_vector = tuple(-np.array(new_y_vector)+0)

            # Flip the origin y component
            new_origin_indices = (new_origin_indices[0], input_image.GetSize()[1]-1, new_origin_indices[2])
            image_modified = True

        # Is the greatest z pixel index physically superior or inferior?
        # Greatest physical position is superior
        if last_index_location[2] < first_index_location[2]: # The highest z pixel value is inferior - should be superior
            # Flip about the z axis
            converted_image_data = np.flip(converted_image_data, 0)

            # Flip the orientation z component
            new_z_vector = tuple(-np.array(new_z_vector)+0)  

            # Flip the origin z component
            new_origin_indices = (new_origin_indices[0], new_origin_indices[1], input_image.GetSize()[2]-1)

            image_modified = True
        
    elif plane == 'coronal':
        # Is the greatest x pixel index physically left or right?
        # Greatest physical position (biggest number) is left
        if last_index_location[0] < first_index_location[0]: # The highest x pixel value is right - should be left
            # Flip about the x axis
            converted_image_data = np.flip(converted_image_data, 2)

            # Flip the orientation x component
            new_x_vector = tuple(-np.array(new_x_vector)+0)

            # Flip the origin x component
            new_origin_indices = (input_image.GetSize()[0]-1, new_origin_indices[1], new_origin_indices[2])

            image_modified = True

        # Is the greatest y pixel index physically superior or inferior?
        # Greatest physical position is superior
        if last_index_location[2] > first_index_location[2]: # The highest z pixel value is superior - should be inferior
            # Flip about the y axis
            converted_image_data = np.flip(converted_image_data, 1)

            # Flip the orientation y component
            new_y_vector = tuple(-np.array(new_y_vector)+0)

            # Flip the origin y component
            new_origin_indices = (new_origin_indices[0], input_image.GetSize()[1]-1, new_origin_indices[2])

            image_modified = True

        # Is the greatest z pixel index physically posterior or anterior?
        # Greatest physical position is posterior
        if last_index_location[1] < first_index_location[1]: # The highest z pixel value is anterior - should be posterior
            # Flip about the z axis
            converted_image_data = np.flip(converted_image_data, 0)

            # Flip the orientation z component
            new_z_vector = tuple(-np.array(new_z_vector)+0)

            # Flip the origin z component
            new_origin_indices = (new_origin_indices[0], new_origin_indices[1], input_image.GetSize()[2]-1)

            image_modified = True
    elif plane == 'sagittal':
        # Is the greatest x pixel index physically posterior or anterior?
        # Greatest physical position (biggest number) is posterior
        if last_index_location[1] < first_index_location[1]: # The highest x pixel value is anterior - should be posterior
            # Flip about the x axis
            converted_image_data = np.flip(converted_image_data, 2)

            # Flip the orientation x component
            new_x_vector = tuple(-np.array(new_x_vector)+0)

            # Flip the origin x component
            new_origin_indices = (input_image.GetSize()[0]-1, new_origin_indices[1], new_origin_indices[2])

            image_modified = True
        
        # Is the greatest y pixel index physically superior or inferior?
        # Greatest physical position is superior
        if last_index_location[2] > first_index_location[2]: # The highest z pixel value is superior - should be inferior
            # Flip about the y axis
            converted_image_data = np.flip(converted_image_data, 1)

            # Flip the orientation y component
            new_y_vector = tuple(-np.array(new_y_vector)+0)

            # Flip the origin y component
            new_origin_indices = (new_origin_indices[0], input_image.GetSize()[1]-1, new_origin_indices[2])

            image_modified = True

        # Is the greatest z pixel index physically left or right?
        # Greatest physical position is left
        if last_index_location[0] < first_index_location[0]: # The highest z pixel value is right - should be left
            # Flip about the z axis
            converted_image_data = np.flip(converted_image_data, 0)

            # Flip the orientation z component
            new_z_vector = tuple(-np.array(new_z_vector)+0)

            # Flip the origin z component
            new_origin_indices = (new_origin_indices[0], new_origin_indices[1], input_image.GetSize()[2]-1)

            image_modified = True

    if image_modified:
        converted_image = sitk.GetImageFromArray(converted_image_data)
        new_origin = input_image.TransformIndexToPhysicalPoint(new_origin_indices)
        converted_image.SetOrigin(new_origin)

        new_direction = (new_x_vector[0], new_y_vector[0], new_z_vector[0],
                        new_x_vector[1], new_y_vector[1], new_z_vector[1],
                        new_x_vector[2], new_y_vector[2], new_z_vector[2])
        converted_image.SetDirection(new_direction)
        
        converted_image.SetSpacing(input_image.GetSpacing())
    else:
        converted_image = input_image
    
    return converted_image

def predict_low_res_interpolate(input_sitk, model_path, spacing=(1.5,1.5,1.5), roi=(96,96,96), min_voxel_value=None, max_voxel_value=None, num_labels=1, keep_only_largest=True, force_cpu=False, verbose=False):
    """ Same as predict_low_res but the prediction results are interpolated when resampled back to original image resolution.
    Results in a smoother and likely more accurate segmentation.

    Parameters:
        input_sitk (sitk.Image): Loaded sitk image
        model_path (str): Path to model to be used for segmentation prediction
        spacing (list): x,y,z spacing e.g. (1.5,1.5,1.5)
        roi (list): x,y,z matrix size e.g. roi=(96,96,96) - MUST be a multiple of 16
        min_voxel_value (int): smallest possible value of the provided image (need to specify to avoid outliers)
        max_voxel_value (int): largest possible value of the provided image (need to specify to avoid outliers)        
        num_labels (int): Number of labels to predict
        keep_only_largest (bool): Whether to only keep the largest connected component of each label
        force_cpu (bool): If true, forces use of CPU for prediction even if GPU is available.
        verbose (bool): Verbosity flag

    Returns:
        prediction_sitk (sitkImage): Prediction label for the provided image and model
    """ 
    start_time = None
    if verbose:
        start_time = time.time()
        print('Predicting low res with ' + model_path + '..', end='')
    
    # Variables
    x_spacing = spacing[0]
    y_spacing = spacing[1]
    z_spacing = spacing[2]

    if min_voxel_value is None or max_voxel_value is None:
        input_sitk_data = sitk.GetArrayFromImage(input_sitk)
        min_voxel_value = np.min(input_sitk_data)
        max_voxel_value = np.max(input_sitk_data)

    # Prepare the image for prediction
    resampled_sitk = resample_spacing(input_sitk, new_spacing=[x_spacing,y_spacing,z_spacing])
    resampled_data = sitk.GetArrayFromImage(resampled_sitk)
    resampled_rescaled_data = np.interp(resampled_data, (min_voxel_value, max_voxel_value), (0, 1))  
    resampled_rescaled_data = np.transpose(resampled_rescaled_data, (2, 1, 0)) # Transpose from z,x,y to z,y,x (convert from sitk to RAS)
    
    resampled_rescaled_data = np.array(resampled_rescaled_data)[np.newaxis, np.newaxis, :, :, :]
    resampled_rescaled_data = torch.from_numpy(resampled_rescaled_data)
    resampled_rescaled_data = resampled_rescaled_data.type(torch.FloatTensor)

    # Make the prediction
    device = None
    if torch.cuda.is_available() and not force_cpu:
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        
        # For simplicity, use the first GPU (you can have custom logic here)
        # You can also set up multi-GPU training here if desired
        selected_gpu = 0
        if verbose:
            if num_gpus > 1:
                print(f"{num_gpus} GPUs available. Using GPU {selected_gpu}. ")
            else:
                print("Using the available GPU. ")
        
        device = torch.device(f"cuda:{selected_gpu}")
    else:
        # Fallback to CPU
        if verbose:
            print("No GPU available. Using CPU. ")
        device = torch.device("cpu")

    model = monai.networks.nets.UNet(spatial_dims=3, in_channels=1, out_channels=num_labels+1, channels=(16, 32, 64, 128, 256),
                                    strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)                                  
    model.load_state_dict(torch.load(model_path, map_location=device))
    sw_batch_size = 4

    predict_data = sliding_window_inference(resampled_rescaled_data.to(device), roi, sw_batch_size, model, overlap = 0.25)

    # Prepare the image for output (convert to numpy, undo the initial transforms of transpose and resample)
    predict_data = torch.argmax(predict_data, dim=1).detach().cpu().numpy()
    predict_data = predict_data[0,:,:,:]
    predict_data = np.transpose(predict_data, (2, 1, 0)) # Reverse transpose - basically convert back from RAS to sitk
    predict_data = predict_data.astype(np.float32)
    
    predict_sitk = sitk.GetImageFromArray(predict_data)
    predict_sitk.CopyInformation(resampled_sitk)
    orig_spacing = input_sitk.GetSpacing()
    predict_sitk = resample_spacing(predict_sitk, new_spacing=orig_spacing, interpolation=sitk.sitkLinear)
    predict_sitk_data = sitk.GetArrayFromImage(predict_sitk)

    # Odd correction from resampling - sometimes there is a "shell" of num_labels+1 at the periphery - remove that if present
    extra_labels_indices = np.where(predict_sitk_data > num_labels)
    predict_sitk_data[extra_labels_indices] = 0

    # Correct for linear interpolation
    zero_indices = np.where(predict_sitk_data < 0.5)
    one_indices = np.where(predict_sitk_data >= 0.5)
    predict_sitk_data[zero_indices] = 0.0
    predict_sitk_data[one_indices] = 1.0
    predict_sitk_data = predict_sitk_data.astype(np.int16)

    if num_labels == 1 and keep_only_largest:
        # From: https://stackoverflow.com/questions/47520487/how-to-use-python-opencv-to-find-largest-connected-component-in-a-single-channel?rq=1
        predict_sitk_data = keep_only_largest_connected_components(predict_sitk_data)

    # predict_sitk_data = z,y,x
    # input_sitk.GetSize() = x,y,z

    # Odd correction necessary - set the max x, y, z values = 0 (there's a "shell" label at these locations for some reason)
    # Compare x dim
    x_diff = predict_sitk_data.shape[2] - input_sitk.GetSize()[0]
    if x_diff > 0:
        for i in range(x_diff):
            predict_sitk_data = np.delete(predict_sitk_data, predict_sitk_data.shape[2]-1, axis=2)
    else:
        predict_sitk_data[:,:,predict_sitk_data.shape[2]-1] = 0
    if x_diff < 0:
        append_data = np.zeros((predict_sitk_data.shape[0], predict_sitk_data.shape[1], 1), dtype=np.int16)
        for i in range(abs(x_diff)):
            predict_sitk_data = np.append(predict_sitk_data, append_data, axis=2)
    
    # Compare y dim
    y_diff = predict_sitk_data.shape[1] - input_sitk.GetSize()[1]
    if y_diff > 0:
        for i in range(y_diff):
            predict_sitk_data = np.delete(predict_sitk_data, predict_sitk_data.shape[1]-1, axis=1)
    else:
        predict_sitk_data[:,predict_sitk_data.shape[1]-1,:] = 0
    if y_diff < 0:
        append_data = np.zeros((predict_sitk_data.shape[0], 1, predict_sitk_data.shape[2]), dtype=np.int16)
        for i in range(abs(y_diff)):
            predict_sitk_data = np.append(predict_sitk_data, append_data, axis=1)

    # Compare z dim
    z_diff = predict_sitk_data.shape[0] - input_sitk.GetSize()[2]
    if z_diff > 0:
        for i in range(z_diff):
            predict_sitk_data = np.delete(predict_sitk_data, predict_sitk_data.shape[0]-1, axis=0)
    else:
        predict_sitk_data[predict_sitk_data.shape[0]-1,:,:] = 0    
    if z_diff < 0:
        append_data = np.zeros((1, predict_sitk_data.shape[1], predict_sitk_data.shape[2]), dtype=np.int16)
        for i in range(abs(z_diff)):
            predict_sitk_data = np.append(predict_sitk_data, append_data, axis=0)

    corrected_predict_sitk = sitk.GetImageFromArray(predict_sitk_data)
    corrected_predict_sitk.CopyInformation(input_sitk)

    if verbose:
        print('.done - time = ' + str(np.around((time.time() - start_time), decimals=1)) + 'sec') 

    return corrected_predict_sitk


def resample_spacing(input_image, new_spacing=[1,1,1], interpolation=sitk.sitkBSpline, verbose=False):
    """ Resamples to the specified pixel spacing, in mm

    Parameters:
        input_image (SimpleITK.Image): Simple ITK image to be resampled
        new_spacing (list, default = [1,1,1]): New spacing in mm, [x,y,z]
        is_label (Boolean - default = False): Whether to resample the image as anatomic or label
        verbose (bool): Verbose flag

    Returns:
        output_image (SimpleITK.Image): Resampled Simple ITK image
    """
    # This page really helped:
    # https://gist.github.com/mrajchl/ccbd5ed12eb68e0c1afc5da116af614a
    
    start_time = None
    if verbose:
        start_time = time.time()
        print("Beginning resample_spacing..", end = '')

    original_spacing = input_image.GetSpacing()
    original_size = input_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(input_image.GetDirection())
    resample.SetOutputOrigin(input_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(input_image.GetPixelIDValue())
    resample.SetInterpolator(interpolation)

    output_image = resample.Execute(input_image)

    if verbose:
        print('.done - time = ' + str(np.around((time.time() - start_time), decimals=1)) + 'sec')

    return output_image
