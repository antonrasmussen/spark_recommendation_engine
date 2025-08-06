import os
import sys
import argparse

def split_csv(input_file, chunk_size_mb=20):
    """
    Splits a large CSV file into smaller chunks of a specified size.

    Args:
        input_file (str): The path to the large CSV file.
        chunk_size_mb (int): The desired maximum size for each chunk in megabytes.
    """
    # --- 1. Input Validation and Setup ---
    if not os.path.exists(input_file):
        print(f"Error: File not found at '{input_file}'")
        sys.exit(1)

    # Convert chunk size from MB to bytes
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # Check if splitting is necessary
    file_size = os.path.getsize(input_file)
    if file_size <= chunk_size_bytes:
        print(f"File size ({file_size / (1024*1024):.2f} MB) is not larger than the specified chunk size ({chunk_size_mb} MB). No splitting needed.")
        return

    print(f"File size: {file_size / (1024*1024):.2f} MB. Starting split...")

    try:
        # --- 2. Read Header and Prepare for Splitting ---
        with open(input_file, 'r', encoding='utf-8', newline='') as f_in:
            header = f_in.readline()
            
            # If the file is empty or has only a header
            if not header:
                print("Warning: Input file is empty.")
                return

            # --- 3. Process the File and Create Splits ---
            file_count = 1
            current_chunk_size = 0
            output_file = None
            
            # Get the base name and extension for output files
            base_name, extension = os.path.splitext(os.path.basename(input_file))
            
            for line in f_in:
                # If we need to start a new file (first run or chunk size exceeded)
                if output_file is None or current_chunk_size > chunk_size_bytes:
                    if output_file:
                        output_file.close()
                        print(f"  -> Finished writing {output_filename} ({current_chunk_size / (1024*1024):.2f} MB)")
                    
                    # Create new filename like 'original_file_split_1.csv'
                    output_filename = f"{base_name}_split_{file_count}{extension}"
                    print(f"Creating new split: {output_filename}")
                    
                    output_file = open(output_filename, 'w', encoding='utf-8', newline='')
                    
                    # Write header to the new split
                    output_file.write(header)
                    
                    # Reset chunk size, accounting for the header's size
                    current_chunk_size = len(header.encode('utf-8'))
                    file_count += 1
                
                # Write the current line to the active split
                output_file.write(line)
                current_chunk_size += len(line.encode('utf-8'))
            
            # --- 4. Finalize ---
            if output_file:
                output_file.close()
                print(f"  -> Finished writing {output_filename} ({current_chunk_size / (1024*1024):.2f} MB)")

        print("\nSplitting complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a large CSV file into smaller chunks, preserving the header in each.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        help="The path to the large CSV file you want to split."
    )
    
    parser.add_argument(
        "-s", "--size",
        type=int,
        default=20,
        help="The maximum size of each split file in megabytes (MB). Default is 20."
    )

    args = parser.parse_args()
    
    split_csv(args.input_file, args.size)