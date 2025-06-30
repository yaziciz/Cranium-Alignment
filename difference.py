import SimpleITK as sitk
import numpy as np
import pyvista as pv

def get_center_of_mass(image):
    """
    Compute the center of mass of the nonzero region in a 3D image or from a file path.
    If the image is a file path, it will be loaded as a SimpleITK image.
    If the image is empty, returns the center of the array.
    """
    # If input is a file path, load the image
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    arr = sitk.GetArrayFromImage(image)
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        return np.array(arr.shape) / 2
    return coords.mean(axis=0)

def get_principal_axes(image):
    """
    Compute the principal axes (eigenvectors of the covariance matrix)
    of the nonzero region in a 3D image or from a file path.
    If the image is a file path, it will be loaded as a SimpleITK image.
    Returns a 3x3 matrix whose columns are the principal axes.
    If there are fewer than 3 points, returns the identity matrix.
    """
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    arr = sitk.GetArrayFromImage(image)
    coords = np.argwhere(arr > 0)
    if coords.shape[0] < 3:
        return np.eye(3)
    coords = coords - coords.mean(axis=0)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # Sort eigenvectors by descending eigenvalue
    return eigvecs[:, order]
def compute_transform_metrics(vol1, vol2):
    """
    Compute translation and rotation metrics between two volumes.
    - Translation: difference between centers of mass.
    - Rotation: rotation matrix aligning principal axes of vol1 to vol2,
      and the corresponding rotation angle in degrees.
    """
    # Compute centers of mass
    com1 = get_center_of_mass(vol1)
    com2 = get_center_of_mass(vol2)
    translation = com2 - com1

    # Compute principal axes for both volumes
    axes1 = get_principal_axes(vol1)
    axes2 = get_principal_axes(vol2)

    # Compute rotation matrix that aligns axes1 to axes2
    rot = axes2 @ axes1.T
    # Compute the rotation angle from the rotation matrix
    angle = np.arccos((np.trace(rot) - 1) / 2)
    return translation, rot, np.degrees(angle)

def visualize_volumes(vol1, vol2, title="Volume Alignment Difference"):
    """
    Visualize two volumes overlaid using PyVista.
    Volume 1 is shown in red, Volume 2 in blue.
    """
    if isinstance(vol1, str):
        vol1 = sitk.ReadImage(vol1)
    if isinstance(vol2, str):
        vol2 = sitk.ReadImage(vol2)
    arr1 = sitk.GetArrayFromImage(vol1)
    arr2 = sitk.GetArrayFromImage(vol2)
    grid1 = pv.wrap(arr1)
    grid2 = pv.wrap(arr2)
    p = pv.Plotter()
    # Add first volume in red shades
    p.add_volume(grid1, opacity="sigmoid", cmap="Reds", name="Volume 1")
    # Add second volume in blue shades
    p.add_volume(grid2, opacity="sigmoid", cmap="Blues", name="Volume 2")
    p.add_legend([("Volume 1", "red"), ("Volume 2", "blue")])
    p.show(title=title)

def show_difference(vol1, vol2):
    """
    Compute and visualize the difference between two volumes.
    Displays translation vector, rotation matrix, and angle.
    """
    translation, rotation, angle = compute_transform_metrics(vol1, vol2)
    
    print(f"Translation vector: {translation}")
    print(f"Rotation matrix:\n{rotation}")
    print(f"Rotation angle (degrees): {angle}")

    # Visualize the volumes
    visualize_volumes(vol1, vol2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python difference.py <volume1> <volume2>")
        sys.exit(1)
    
    vol1_path = sys.argv[1]
    vol2_path = sys.argv[2]
    
    show_difference(vol1_path, vol2_path)