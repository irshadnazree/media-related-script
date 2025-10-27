import os
import shutil
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# Supported image extensions (case-insensitive)
SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".ico",
)


def get_bool_input(prompt_message):
    """Gets a yes/no answer from the user."""
    while True:
        choice = input(f"{prompt_message} (y/n): ").strip().lower()
        if choice in ["y", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def process_image(
    input_path,
    output_path,
    do_compress,
    compression_quality,
    do_convert,
    convert_format,
    keep_resolution=True,
):
    """Processes a single image: resizes if needed, converts first, then compresses."""
    try:
        img = Image.open(input_path)
        original_format = img.format

        # Step 0: Handle resizing if compression is enabled and keep_resolution is False
        if do_compress and not keep_resolution and img.height > 1600:
            # Store original dimensions for logging
            original_width, original_height = img.width, img.height
            
            # Calculate new dimensions while maintaining aspect ratio
            aspect_ratio = img.width / img.height
            new_height = 1600
            new_width = int(new_height * aspect_ratio)
            
            # Resize the image using high-quality resampling
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")

        # Step 1: Handle conversion if needed
        if do_convert:
            # Handle transparency when converting to formats that don't support it (like JPEG)
            if convert_format.lower() == "jpeg":
                if img.mode in ("RGBA", "LA", "P"):
                    # Create a white background image
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    # Paste the image onto the background, using alpha channel as mask
                    background.paste(img, (0, 0), img.convert("RGBA"))
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")
            elif convert_format.lower() == "ico":
                # ICO format requirements: RGBA mode and size constraints
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                # ICO format works best with square images and common icon sizes
                # If image is not square, we'll keep it as is but warn the user
                if img.width != img.height:
                    print(f"Warning: ICO format works best with square images. Current size: {img.width}x{img.height}")
            elif convert_format.lower() == "png":
                if img.mode != "RGBA" and img.mode != "RGB": # Keep RGBA if already, else RGB
                    img = img.convert("RGBA" if "A" in img.mode else "RGB")
            else:
                # For other formats, ensure proper mode conversion
                if img.mode not in ["RGB", "RGBA"]:
                    img = img.convert("RGBA" if "A" in img.mode else "RGB")

        # Step 2: Handle compression if needed
        save_options = {}
        if do_compress:
            if (
                (do_convert and convert_format.lower() in ["jpeg", "webp"])
                or (not do_convert and original_format.lower() in ["jpeg", "webp"])
            ):
                save_options["quality"] = compression_quality
            elif (
                (do_convert and convert_format.lower() == "png")
                or (not do_convert and original_format.lower() == "png")
            ):
                save_options["optimize"] = True

        # Determine the format for saving
        save_format = convert_format if do_convert else original_format

        # Save the processed image
        img.save(output_path, format=save_format, **save_options)
        return True

    except FileNotFoundError:
        print(f"Error: Input image not found at {input_path}")
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {input_path}. It might be corrupted or not a supported image.")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
    return False


def process_images_concurrently(
    image_files_to_process,
    output_dir_path,
    do_compress,
    compression_quality,
    do_convert,
    convert_format,
    do_rename,
    base_rename_name,
    keep_resolution=True,
):
    """Process multiple images concurrently using ThreadPoolExecutor."""
    processed_count = 0
    failed_count = 0
    
    # Calculate optimal number of workers (threads)
    # Use number of CPU cores * 2 for I/O bound tasks
    max_workers = min(multiprocessing.cpu_count() * 2, len(image_files_to_process))
    
    print(f"\nProcessing images using {max_workers} workers...")
    
    def process_single_image(args):
        i, image_file_path = args
        try:
            original_basename, original_ext = os.path.splitext(
                os.path.basename(image_file_path)
            )

            # Determine new name
            if do_rename:
                current_base_name = base_rename_name if base_rename_name else original_basename
                if len(image_files_to_process) > 1:
                    new_name_part = f"{current_base_name}-{i + 1}"
                else:
                    new_name_part = current_base_name
            else:
                new_name_part = original_basename

            # Determine new extension
            if do_convert:
                new_extension = f".{convert_format.lower()}"
            else:
                new_extension = original_ext.lower()

            # For single files, add suffix to avoid overwriting original unless renamed or converted
            if len(image_files_to_process) == 1 and not do_rename and not do_convert:
                # Add "_processed" suffix to avoid overwriting the original file
                new_name_part = f"{new_name_part}_processed"

            output_filename = f"{new_name_part}{new_extension}"
            output_image_path = os.path.join(output_dir_path, output_filename)

            success = process_image(
                image_file_path,
                output_image_path,
                do_compress,
                compression_quality,
                do_convert,
                convert_format,
                keep_resolution,
            )
            return success
        except Exception as e:
            print(f"\nError processing {image_file_path}: {e}")
            return False

    # Create a list of (index, file_path) tuples for processing
    processing_args = list(enumerate(image_files_to_process))
    
    # Process images concurrently with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and get futures
        future_to_image = {
            executor.submit(process_single_image, args): args[1]
            for args in processing_args
        }
        
        # Process results as they complete with progress bar
        with tqdm(total=len(image_files_to_process), desc="Progress", unit="image") as pbar:
            for future in concurrent.futures.as_completed(future_to_image):
                if future.result():
                    processed_count += 1
                else:
                    failed_count += 1
                pbar.update(1)
    
    return processed_count, failed_count


def main():
    print("--- Image Processing Script ---")

    while True:
        input_path = input("Enter the path to an image file or a directory: ").strip()
        
        # Handle paths with spaces by stripping quotes and normalizing
        if input_path.startswith('"') and input_path.endswith('"'):
            input_path = input_path[1:-1]
        elif input_path.startswith("'") and input_path.endswith("'"):
            input_path = input_path[1:-1]
        else:
            # Handle escaped spaces (e.g., "test\ name.webp" -> "test name.webp")
            input_path = input_path.replace('\\ ', ' ')
        
        # Expand user home directory (~) and normalize path
        input_path = os.path.expanduser(input_path)
        input_path = os.path.normpath(input_path)
        
        if os.path.exists(input_path):
            break
        else:
            print("Error: The specified path does not exist. Please try again.")
            print("Tip: If your path contains spaces, you can wrap it in quotes like: \"/path/with spaces/\"")

    # --- Get user choices for operations ---
    # Enhanced format conversion input - allows multi-parameter input
    print("\nConvert image format?:")
    print("Format: 'y [format_number]/n'")
    print("Available formats: 1=JPEG, 2=PNG, 3=WEBP, 4=GIF, 5=TIFF, 6=BMP, 7=ICO")
    
    while True:
        convert_input = input("Enter conversion settings: ").strip().lower()
        parts = convert_input.split()
        
        if not parts:
            print("Invalid input. Please enter at least 'yes' or 'no'.")
            continue
            
        # Parse the first parameter (convert yes/no)
        if parts[0] in ["Y", "y", "yes"]:
            do_convert = True
        elif parts[0] in ["N", "n", "no"]:
            do_convert = False
            convert_format = "JPEG"  # Default (won't be used)
            break
        else:
            print("Invalid input. Please start with 'yes' or 'no'.")
            continue
        
        # Define available formats with their corresponding numbers
        format_options = {
            "1": "JPEG",
            "2": "PNG", 
            "3": "WEBP",
            "4": "GIF",
            "5": "TIFF",
            "6": "BMP",
            "7": "ICO"
        }
        
        # Set default format
        convert_format = "JPEG"
        
        # Parse additional parameters if provided
        if len(parts) >= 2:
            # Second parameter: format number
            format_choice = parts[1]
            if format_choice in format_options:
                convert_format = format_options[format_choice]
            else:
                print(f"Invalid format choice '{format_choice}'. Using default (JPEG).")
                print("Available formats: 1=JPEG, 2=PNG, 3=WEBP, 4=GIF, 5=TIFF, 6=BMP, 7=ICO")
        
        break

    # Enhanced compression input - allows multi-parameter input
    print("\nCompress images?:")
    print("Format: 'y [quality] [allow_downsizing]/n'")
    
    while True:
        compress_input = input("Enter compression settings: ").strip().lower()
        parts = compress_input.split()
        
        if not parts:
            print("Invalid input. Please enter at least 'yes' or 'no'.")
            continue
            
        # Parse the first parameter (compress yes/no)
        if parts[0] in ["Y", "y", "yes"]:
            do_compress = True
        elif parts[0] in ["N", "n", "no"]:
            do_compress = False
            compression_quality = 85  # Default (won't be used)
            keep_resolution = True  # Default (won't be used)
            break
        else:
            print("Invalid input. Please start with 'yes' or 'no'.")
            continue
        
        # Set defaults
        compression_quality = 85
        allow_downsizing = True  # Default is to allow downsizing
        
        # Parse additional parameters if provided
        if len(parts) >= 2:
            # Second parameter: quality
            try:
                quality = int(parts[1])
                if 1 <= quality <= 100:
                    compression_quality = quality
                else:
                    print("Quality must be between 1 and 100. Using default (85).")
            except ValueError:
                print(f"Invalid quality '{parts[1]}'. Using default (85).")
        
        if len(parts) >= 3:
            # Third parameter: allow downsizing
            if parts[2] in ["y", "yes"]:
                allow_downsizing = True
            elif parts[2] in ["n", "no"]:
                allow_downsizing = False
            else:
                print(f"Invalid downsizing option '{parts[2]}'. Using default (yes).")
        
        keep_resolution = not allow_downsizing
        break

    do_rename = get_bool_input("Rename images?")
    base_rename_name = ""
    if do_rename:
        base_rename_name = input(
            "Enter the base name for renaming (e.g., 'holiday_image'): "
        ).strip()
        if not base_rename_name:
            print("No base name provided, renaming will use original base names if not a batch.")

    # --- Prepare output directory ---
    if os.path.isfile(input_path):
        # For single files, save in the same directory as the input file
        output_dir_path = os.path.dirname(input_path)
        print(f"Processed file will be saved in: {output_dir_path}")
    else:
        # For directories, create a new subdirectory
        input_dir = input_path
        output_folder_name = os.path.basename(os.path.normpath(input_path)) + "_processed"
        output_dir_path = os.path.join(input_dir, output_folder_name)
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Processed files will be saved in: {output_dir_path}")

    # --- Collect image files ---
    image_files_to_process = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith(SUPPORTED_EXTENSIONS):
            image_files_to_process.append(input_path)
        else:
            print(f"Skipping non-image or unsupported file: {input_path}")
    elif os.path.isdir(input_path):
        for item in os.listdir(input_path):
            item_path = os.path.join(input_path, item)
            if os.path.isfile(item_path) and item.lower().endswith(
                SUPPORTED_EXTENSIONS
            ):
                image_files_to_process.append(item_path)
    else:
        print(f"Error: Input path {input_path} is neither a file nor a directory.")
        return

    if not image_files_to_process:
        print("No image files found to process.")
        return

    # --- Process images concurrently ---
    processed_count, failed_count = process_images_concurrently(
        image_files_to_process,
        output_dir_path,
        do_compress,
        compression_quality,
        do_convert,
        convert_format,
        do_rename,
        base_rename_name,
        keep_resolution,
    )

    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count} images")
    print(f"Failed to process: {failed_count} images")
    print(f"Total images: {len(image_files_to_process)}")
    print(f"Output directory: {output_dir_path}")


if __name__ == "__main__":
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        print("Pillow library is not installed. Please install it by running:")
        print("pip install Pillow")
        exit()
    main()
