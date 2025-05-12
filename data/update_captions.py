import csv
import os
import tempfile # Used for safely creating a temporary file
import shutil   # Used for safely replacing the original file

# --- Configuration ---
CAPTIONS_FILE_TO_UPDATE = "data/captions.txt"  # The file that WILL BE MODIFIED
TEST_CAPTIONS_FILE = "data/captions_test.txt" # File defining which images to remove
# ---

print(f"WARNING: This script will modify '{CAPTIONS_FILE_TO_UPDATE}' in place.")
print("Make sure you have a backup if needed.")
proceed = input("Continue? (yes/no): ").strip().lower()
if proceed != 'yes':
    print("Operation cancelled.")
    exit()

print(f"\nIdentifying test image filenames from '{TEST_CAPTIONS_FILE}'...")
test_image_filenames = set()

# --- Step 1: Read TEST file to find unique image filenames in the test set ---
try:
    with open(TEST_CAPTIONS_FILE, 'r', newline='', encoding='utf-8') as infile_test:
        reader_test = csv.reader(infile_test)
        try:
            _ = next(reader_test) # Read and discard the header row
            for row in reader_test:
                if row: # Check if row is not empty
                    image_filename = row[0] # Assumes first column is filename
                    test_image_filenames.add(image_filename)
        except StopIteration:
            print(f"Warning: Test captions file '{TEST_CAPTIONS_FILE}' is empty or contains only a header.")
            # Continue, as no files will be removed if test set is empty
except FileNotFoundError:
    print(f"Error: Test captions file '{TEST_CAPTIONS_FILE}' not found. Cannot determine which captions to remove.")
    exit()
except Exception as e:
    print(f"An error occurred while reading {TEST_CAPTIONS_FILE}: {e}")
    exit()

if not test_image_filenames:
    print("No test images found in test file. No changes will be made.")
    exit()

print(f"Found {len(test_image_filenames)} unique test image filenames to remove from '{CAPTIONS_FILE_TO_UPDATE}'.")

# --- Step 2: Write filtered data (excluding test images) to a temporary file ---
temp_file_path = None
original_header = None
lines_kept_count = 0
lines_removed_count = 0
error_occurred = False

try:
    # Create a temporary file securely in the same directory as the original
    # This makes os.replace (shutil.move) more likely to be an atomic operation
    file_dir = os.path.dirname(os.path.abspath(CAPTIONS_FILE_TO_UPDATE)) or '.'
    with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8', delete=False, dir=file_dir, suffix='.tmp') as temp_outfile:
        temp_file_path = temp_outfile.name
        writer_temp = csv.writer(temp_outfile)

        with open(CAPTIONS_FILE_TO_UPDATE, 'r', newline='', encoding='utf-8') as infile_original:
            reader_original = csv.reader(infile_original)

            # Read and write the header
            try:
                original_header = next(reader_original)
                writer_temp.writerow(original_header)
            except StopIteration:
                print(f"Error: Original file '{CAPTIONS_FILE_TO_UPDATE}' is empty or has no header.")
                raise # Re-raise to trigger cleanup in except block

            # Process data rows
            for row in reader_original:
                if row:
                    image_filename = row[0]
                    if image_filename not in test_image_filenames:
                        writer_temp.writerow(row)
                        lines_kept_count += 1
                    else:
                        lines_removed_count += 1

except FileNotFoundError:
    print(f"Error: Original captions file '{CAPTIONS_FILE_TO_UPDATE}' not found during processing.")
    error_occurred = True
except Exception as e:
    print(f"An error occurred during processing or writing to temporary file: {e}")
    error_occurred = True
finally:
    # Ensure temporary file is closed if open, although 'with' handles this
    pass # 'with' statement handles closing temp_outfile

# --- Step 3: Replace original file with temporary file if no errors occurred ---
if not error_occurred and temp_file_path:
    try:
        # shutil.move is generally preferred for replacing files across different systems/mounts
        # os.replace is more atomic on POSIX systems if src/dst are on the same filesystem
        shutil.move(temp_file_path, CAPTIONS_FILE_TO_UPDATE)
        print("\nFinished processing.")
        print(f"Successfully updated '{CAPTIONS_FILE_TO_UPDATE}'.")
        print(f"  Lines kept (excluding header): {lines_kept_count}")
        print(f"  Lines removed: {lines_removed_count} (corresponding to {len(test_image_filenames)} unique test images)")
    except Exception as e:
        print(f"\nError: Failed to replace original file with temporary file.")
        print(f"  The original file '{CAPTIONS_FILE_TO_UPDATE}' has NOT been modified.")
        print(f"  The filtered content might be found in '{temp_file_path}' (if not cleaned up).")
        print(f"  Error details: {e}")
        error_occurred = True # Mark error to prevent cleaning temp file if user wants to inspect it
else:
    # An error occurred during the reading/writing phase
    print("\nProcessing failed. Original file was not modified.")

# Clean up temporary file if an error occurred and it still exists, unless replacement failed
if error_occurred and temp_file_path and os.path.exists(temp_file_path):
    # Optionally keep the temp file on error for inspection
    # print(f"Temporary file kept at: {temp_file_path}")
    # Or remove it:
    try:
        os.remove(temp_file_path)
        print("Temporary file cleaned up after error.")
    except OSError as ose:
        print(f"Warning: Could not clean up temporary file '{temp_file_path}': {ose}")
