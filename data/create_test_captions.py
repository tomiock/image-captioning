import csv
import random
import os

# --- Configuration ---
INPUT_CAPTIONS_FILE = "data/captions.txt"
OUTPUT_TEST_FILE = "data/captions_test.txt"  # Only this file will be created
SAMPLE_PERCENTAGE = 0.05  # 5% for the test set
# ---

print(f"Reading unique image filenames from '{INPUT_CAPTIONS_FILE}'...")

unique_image_filenames = set()
header = None

# --- Step 1: Read input file to find unique image filenames ---
try:
    with open(INPUT_CAPTIONS_FILE, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        try:
            header = next(reader)  # Read and store the header row
            print(f"Detected header: {header}")
            for row in reader:
                if row:  # Check if row is not empty
                    image_filename = row[0]  # First column is the image filename
                    unique_image_filenames.add(image_filename)
        except StopIteration:
            print(f"Error: Input file '{INPUT_CAPTIONS_FILE}' is empty or contains only a header.")
            exit()
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_CAPTIONS_FILE}' not found.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the input file: {e}")
    exit()

if not unique_image_filenames:
    print("No image filenames found after the header.")
    exit()

# --- Step 2: Sample unique filenames for the test set ---
total_unique_images = len(unique_image_filenames)
num_to_sample = int(total_unique_images * SAMPLE_PERCENTAGE)

# Ensure at least one image is sampled if possible and percentage > 0
if num_to_sample == 0 and total_unique_images > 0 and SAMPLE_PERCENTAGE > 0:
    print(f"Calculated sample size is 0 (5% of {total_unique_images}). Selecting 1 unique image instead.")
    num_to_sample = 1
elif num_to_sample > total_unique_images:
    num_to_sample = total_unique_images # Cannot sample more than available


print(f"Found {total_unique_images} unique image filenames.")
print(f"Sampling {num_to_sample} unique image filenames for the test set.")

# Perform the random sampling
test_image_filenames = set(random.sample(list(unique_image_filenames), num_to_sample))
print(f"Selected {len(test_image_filenames)} filenames for '{OUTPUT_TEST_FILE}'.")

# --- Step 3: Read input file again and write ONLY the test file ---
print(f"Writing test captions to '{OUTPUT_TEST_FILE}'...")
test_lines_written = 0

try:
    with open(INPUT_CAPTIONS_FILE, 'r', newline='', encoding='utf-8') as infile, \
         open(OUTPUT_TEST_FILE, 'w', newline='', encoding='utf-8') as outfile_test:

        reader = csv.reader(infile)
        writer_test = csv.writer(outfile_test)

        # Write the header to the test output file
        if header:
            writer_test.writerow(header)
        else:
             print("Warning: Header was not read correctly (should not happen).")

        next(reader)  # Skip the header row in the input file for data processing

        # Write only the rows corresponding to test images
        for row in reader:
            if row:  # Check if row is not empty
                image_filename = row[0]
                if image_filename in test_image_filenames:
                    writer_test.writerow(row)
                    test_lines_written += 1

except Exception as e:
    print(f"An error occurred during writing to {OUTPUT_TEST_FILE}: {e}")
    exit()

print("\nFinished processing.")
print(f"  Total lines written to '{OUTPUT_TEST_FILE}': {test_lines_written} (corresponding to {len(test_image_filenames)} unique images)")
print(f"Original file '{INPUT_CAPTIONS_FILE}' remains unchanged.")
