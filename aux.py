import numpy as np
import SimpleITK as sitk
import pyvista as pv
from scipy.ndimage import center_of_mass, label

MIN_LANDMARK_DISTANCE = 10  # Minimum distance between landmarks in mm


def center_scan(input_image):
    """
    Centers the scan by adjusting the origin so that the physical center of the image aligns with (0, 0, 0).

    Parameters:
        input_image (sitk.Image): The input SimpleITK image.

    Returns:
        sitk.Image: The centered SimpleITK image.
    """
    # Get the size and spacing of the image
    size = np.array(input_image.GetSize())
    spacing = np.array(input_image.GetSpacing())

    # Calculate the physical center of the image
    physical_center = size * spacing / 2.0

    # Set the new origin to align the physical center with (0, 0, 0)
    new_origin = tuple(-physical_center)
    input_image.SetOrigin(new_origin)

    return input_image

def visualize_sitk_image_3d(sitk_image, threshold=0, title = "3D Visualization of the Original Image"):
    """
    Visualizes a SimpleITK image in 3D using PyVista.

    Parameters:
        sitk_image (sitk.Image): The SimpleITK image to visualize.
        threshold (int): Threshold value to create a surface (default is 0).
    """
    # Convert the SimpleITK image to a NumPy array
    image_array = sitk.GetArrayFromImage(sitk_image)  # z, y, x format

    # Create a PyVista dataset from the NumPy array
    grid = pv.wrap(image_array)

    # Apply a threshold to extract the surface
    surface = grid.threshold(value=threshold)

    # Visualize the surface
    plotter = pv.Plotter()
    plotter.add_mesh(surface, opacity=0.5, color="white")
    plotter.add_axes()
    #add title
    plotter.show(title="3D Visualization of the Original Image")

def validate_landmarks(nb_label_image: sitk.Image, rc_label_image: sitk.Image, lc_label_image: sitk.Image):
    """
    Validates the predicted landmarks by checking if they are sufficiently far apart.
    
    Parameters:
        nb_label_image (sitk.Image): Nasion bridge landmark image
        rc_label_image (sitk.Image): Right canthus landmark image
        lc_label_image (sitk.Image): Left canthus landmark image
        
    Returns:
        bool: True if landmarks are valid (sufficiently separated), False otherwise
        nb_center: Center of mass coordinates for nasion bridge landmark
        rc_center: Center of mass coordinates for right canthus landmark 
        lc_center: Center of mass coordinates for left canthus landmark
    """

    #converting into arrays
    nb_label_image_data = sitk.GetArrayFromImage(nb_label_image)
    rc_label_image_data = sitk.GetArrayFromImage(rc_label_image)
    lc_label_image_data = sitk.GetArrayFromImage(lc_label_image)

    #get the mass centers for each landmark

    if np.count_nonzero(nb_label_image_data) == 0 or np.count_nonzero(rc_label_image_data) == 0 or np.count_nonzero(lc_label_image_data) == 0:
        print("One or more landmarks are not found.")
        return False, None, None, None
    
    nb_center = center_of_mass(nb_label_image_data)
    rc_center = center_of_mass(rc_label_image_data)
    lc_center = center_of_mass(lc_label_image_data)
    
    # Calculate distances between landmarks
    nb_rc_distance = np.linalg.norm(np.array(nb_center) - np.array(rc_center))
    nb_lc_distance = np.linalg.norm(np.array(nb_center) - np.array(lc_center))
    rc_lc_distance = np.linalg.norm(np.array(rc_center) - np.array(lc_center))

    #The distances between all the landmarks should be largen than 10mm (experimentally determined)
    if nb_rc_distance < MIN_LANDMARK_DISTANCE or nb_lc_distance < MIN_LANDMARK_DISTANCE or rc_lc_distance < MIN_LANDMARK_DISTANCE:
        print("Mispredicted landmarks detected!")
        if nb_rc_distance < 10:
            print(f"NB-RC {nb_rc_distance:.2f} mm.")
        if nb_lc_distance < 10:
            print(f"NB-LC {nb_lc_distance:.2f} mm.")
        if rc_lc_distance < 10:
            print(f"RC-LC {rc_lc_distance:.2f} mm.")

        return False, nb_center, rc_center, lc_center
    else: return True, nb_center, rc_center, lc_center
    
def trim_skull_as_sphere(input_image: sitk.Image, nb_center, rc_center, lc_center):
    """
    Trims the skull region as a sphere from the input imag using the mispredicted landmarks and threshold.

    Parameters:
        input_image (sitk.Image): The input SimpleITK image.
        threshold (int): Threshold value to trim the skull (default is 0).

    Returns:
        sitk.Image: The trimmed SimpleITK image.
    """
    # Convert the SimpleITK image to a NumPy array
    print("Trimming the skull region...")
    image_array = sitk.GetArrayFromImage(input_image)  # z, y, x format

    # Calculate distances between pairs of landmarks
    landmark_pairs = [
        (nb_center, rc_center),
        (nb_center, lc_center),
        (rc_center, lc_center)
    ]

    distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in landmark_pairs]
    closest_pair_idx = np.argmin(distances)
    closest_combination = landmark_pairs[closest_pair_idx]

    landmark_p1, landmark_p2 = closest_combination
    landmark_p1 = tuple(map(int, landmark_p1))
    landmark_p2 = tuple(map(int, landmark_p2))

    intensity_p1 = image_array[landmark_p1]
    intensity_p2 = image_array[landmark_p2]

    #trim intput starting 1000 until mean of the two landmark intensities
    threshold_high = (intensity_p1 + intensity_p2) / 2
    threshold_low = 1000 #bone lower threshold, https://radiopaedia.org/articles/hounsfield-unit
    trimmed_image_array = np.where((image_array >= threshold_low) & (image_array <= threshold_high), image_array, 0)
    #dilate the trimmed image to ensure the skull is included
    trimmed_image = sitk.GetImageFromArray(trimmed_image_array)

    labeled_array, num_features = label(trimmed_image_array)
    if num_features > 0:
        # Find the largest component (ignore background label 0)
        largest_cc = (labeled_array == (np.bincount(labeled_array.flat)[1:].argmax() + 1))
        trimmed_image_array = trimmed_image_array * largest_cc
        trimmed_image = sitk.GetImageFromArray(trimmed_image_array)

    # Center of mass (in z, y, x)
    center = np.array(center_of_mass(largest_cc))

    # Estimate radius as the max distance from center to any voxel in the component
    largest_cc = sitk.GetImageFromArray(largest_cc.astype(np.uint8))
    largest_cc = sitk.BinaryDilate(largest_cc, [11, 11, 11])  # Adjust the size as needed
    largest_cc = sitk.GetArrayFromImage(largest_cc)
    coords = np.argwhere(largest_cc)

    #largest_cc_image = sitk.GetImageFromArray(largest_cc.astype(np.uint8))
    #sitk.WriteImage(largest_cc_image, "largest_cc.nii.gz")

    if coords.size > 0:
        dists = np.linalg.norm(coords - center, axis=1)
        radius = np.max(dists)

        # Create a spherical mask
        zz, yy, xx = np.ogrid[:largest_cc.shape[0], :largest_cc.shape[1], :largest_cc.shape[2]]
        sphere_mask = ((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2) <= radius**2

        # Keep original values inside the sphere, zero elsewhere
        sphere_image_array = np.where(sphere_mask, image_array, -1000)
        trimmed_image = sitk.GetImageFromArray(sphere_image_array)
        #copy the metadata from the input image
        trimmed_image.CopyInformation(input_image)
        trimmed_image.SetSpacing(input_image.GetSpacing())
        trimmed_image.SetOrigin(input_image.GetOrigin())
        trimmed_image.SetDirection(input_image.GetDirection())

        #sitk.WriteImage(trimmed_image, "trimmed_image.nii.gz")

    else:
        print("No components found in the trimmed image. Returning original image.")
        trimmed_image = input_image

    return trimmed_image

def trim_outlier_region(input_image: sitk.Image, nb_center, rc_center, lc_center):
    """
    Trims the outlier region with a sphere from the input image using the mispredicted landmarks.

    Parameters:
        input_image (sitk.Image): The input SimpleITK image.
        threshold (int): Threshold value to trim the skull (default is 0).

    Returns:
        sitk.Image: The trimmed SimpleITK image.
    """
    # Convert the SimpleITK image to a NumPy array
    print("Removing the outlier region...")
    image_array = sitk.GetArrayFromImage(input_image)  # z, y, x format

    # Calculate distances between pairs of landmarks
    landmark_pairs = [
        (nb_center, rc_center),
        (nb_center, lc_center),
        (rc_center, lc_center)
    ]

    distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in landmark_pairs]
    closest_pair_idx = np.argmin(distances)
    closest_combination = landmark_pairs[closest_pair_idx]

    landmark_p1, landmark_p2 = closest_combination
    landmark_p1 = tuple(map(int, landmark_p1))
    landmark_p2 = tuple(map(int, landmark_p2))

    #take the middle location and create a shpere by a threshold as radius
    middle_location = tuple((np.array(landmark_p1) + np.array(landmark_p2)) // 2)
    
    #create a sphere mask
    radius = 50  # Adjust the radius as needed
    zz, yy, xx = np.ogrid[:image_array.shape[0], :image_array.shape[1], :image_array.shape[2]]
    sphere_mask = ((zz - middle_location[0])**2 + (yy - middle_location[1])**2 + (xx - middle_location[2])**2) <= radius**2
    # zero inside, keep original values outside
    outlier_region_array = np.where(sphere_mask, -1000, image_array)
    trimmed_image = sitk.GetImageFromArray(outlier_region_array)

    #save
    sitk.WriteImage(trimmed_image, "trimmed_outlier_region.nii.gz")
    return trimmed_image

    

    