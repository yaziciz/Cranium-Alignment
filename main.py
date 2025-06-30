import configparser
import glob
import os
import shutil
from difference import show_difference

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import align

DEBUG = True  # Set to True to enable debug mode

def test_align(test_data_path, output_path):
    test_case_paths = sorted(glob.iglob(os.path.join(test_data_path, '*.nii.gz')))
    for test_case_path in test_case_paths:
        filename_full = os.path.basename(test_case_path)
        filename_base = os.path.splitext(os.path.splitext(filename_full)[0])[0]
        aligned_filename = filename_base + '_aligned' + '.nii.gz'
        aligned_path = os.path.join(output_path, aligned_filename)

        print(f"Aligning: {filename_full}")
        align(test_case_path, aligned_path, debug=DEBUG)

        if DEBUG:
            # Show the difference between the original and aligned volumes
            show_difference(test_case_path, aligned_path)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_root, "tests", "outputs")
    test_working_path = os.path.join(project_root, "tests", "test_data")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    test_align(test_working_path, output_path)

