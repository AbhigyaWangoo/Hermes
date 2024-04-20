import os
import shutil

DATA_DIR = "data"


def clear_directory(directory_path: str) -> None:
    """
    Clears out all files and subdirectories within the specified directory.
    """
    # Iterate over all files and directories in the specified directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if the current item is a file
        if os.path.isfile(file_path):
            os.remove(file_path)  # Remove the file
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Remove the directory and its contents recursively
