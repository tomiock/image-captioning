import csv
import os
import shutil

# --- Configuration ---
TEST_CAPTIONS_FILE = "data/captions_test.txt" # File listing test images (in first column)
SOURCE_IMAGE_DIR = "data/Images"          # Directory where all images currently are
DEST_TEST_IMAGE_DIR = "data/test_images"      # Directory where test images will be MOVED
# ---

print(f"Identifying test image filenames from '{TEST_CAPTIONS_FILE}'...")

test_image_filenames = set()

# --- Step 1: Read TEST captions file to find unique image filenames ---
try:
    with open(TEST_CAPTIONS_FILE, 'r', newline='', encoding='utf-8') as infile_test:
        reader_test = csv.reader(infile_test)
        try:
            _ = next(reader_test) # Read and discard the header row
            for row in reader_test:
                if row: # Check if row is not empty
                    image_filename = row[0].strip() # Get filename, remove leading/trailing whitespace
                    if image_filename: # Ensure filename is not empty
                        test_image_filenames.add(image_filename)
        except StopIteration:
            print(f"Warning: Test captions file '{TEST_CAPTIONS_FILE}' appears to be empty or only contains a header.")
except FileNotFoundError:
    print(f"Error: Test captions file '{TEST_CAPTIONS_FILE}' not found. Cannot determine which images to move.")
    exit()
except Exception as e:
    print(f"An error occurred while reading {TEST_CAPTIONS_FILE}: {e}")
    exit()

if not test_image_filenames:
    print("No valid test image filenames found in the captions file. No images will be moved.")
    exit()

print(f"Found {len(test_image_filenames)} unique test image filenames listed in '{TEST_CAPTIONS_FILE}'.")

# --- Step 2: Create destination directory ---
try:
    # exist_ok=True prevents error if the directory already exists
    os.makedirs(DEST_TEST_IMAGE_DIR, exist_ok=True)
    print(f"Ensured destination directory '{DEST_TEST_IMAGE_DIR}' exists.")
except OSError as e:
    print(f"Error: Could not create destination directory '{DEST_TEST_IMAGE_DIR}': {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred creating directory: {e}")
    exit()


# --- Step 3: Move the identified image files ---
print(f"Attempting to move images from '{SOURCE_IMAGE_DIR}' to '{DEST_TEST_IMAGE_DIR}'...")
moved_count = 0
not_found_count = 0
error_count = 0

for filename in test_image_filenames:
    source_path = os.path.join(SOURCE_IMAGE_DIR, filename)
    dest_path = os.path.join(DEST_TEST_IMAGE_DIR, filename)

    # Check if the source file exists before attempting to move
    if os.path.isfile(source_path):
        try:
            # Move the file
            shutil.move(source_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"  ERROR moving '{filename}': {e}")
            error_count += 1
    else:
        # Report if a listed file wasn't found in the source directory
        print(f"  WARNING: Image file '{filename}' listed in test captions not found in '{SOURCE_IMAGE_DIR}'.")
        not_found_count += 1

print("\nFinished moving process.")
print(f"  Successfully moved: {moved_count}")
if not_found_count > 0:
    print(f"  Files listed but not found in source: {not_found_count}")
if error_count > 0:
    print(f"  Errors during move operation: {error_count}")

print(f"\nCheck '{DEST_TEST_IMAGE_DIR}' for the moved test images.")
print(f"'{SOURCE_IMAGE_DIR}' should no longer contain the {moved_count} successfully moved files.")
