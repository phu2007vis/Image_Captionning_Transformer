import os
import shutil

def merge_folders(source1, source2, destination):
    """
    Merge two structured datasets (root1 and root2) into a single destination while preserving hierarchy.
    """

    def copy_files(src, dest):
        for root, _, files in os.walk(src):
            relative_path = os.path.relpath(root, src)
            dest_path = os.path.join(dest, relative_path)
            os.makedirs(dest_path, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, file)

                if os.path.exists(dest_file):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(os.path.join(dest_path, f"{base}_{counter}{ext}")):
                        counter += 1
                    dest_file = os.path.join(dest_path, f"{base}_{counter}{ext}")

                shutil.copy2(src_file, dest_file)

    # Copy both folder structures
    copy_files(source1, destination)
    copy_files(source2, destination)

    print(f"Merged '{source1}' and '{source2}' into '{destination}'")

# Example Usage:
merge_folders("/work/21013187/phuoc/Image_Captionning_Transformer/data/split_ted2",
              "/work/21013187/phuoc/Image_Captionning_Transformer/data/plate_xemay_ver2_splited",
              "/work/21013187/phuoc/Image_Captionning_Transformer/data/split_ted4")
