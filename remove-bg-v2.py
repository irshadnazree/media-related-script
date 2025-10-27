import argparse
import os
import platform
from pathlib import Path
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Global session for worker processes ---
# This avoids re-initializing the model for every image in a single process.
# Each process in the pool will have its own session_worker.
session_worker = None

def init_worker(model_name_init, providers_init):
    """Initializer for each worker process in the pool."""
    global session_worker
    try:
        session_worker = new_session(
            model_name=model_name_init, providers=providers_init
        )
        # print(f"Worker PID {os.getpid()} initialized session with providers: {providers_init}")
    except Exception as e:
        print(
            f"Error initializing session in worker PID {os.getpid()}: {e}"
        )
        # Fallback if CoreML provider fails for some reason in a worker
        if "CoreMLExecutionProvider" in providers_init:
            print(
                f"Worker PID {os.getpid()}: Falling back to CPU for session."
            )
            try:
                session_worker = new_session(
                    model_name=model_name_init,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e_cpu:
                print(
                    f"Critical error: Worker PID {os.getpid()} failed to initialize session even with CPU: {e_cpu}"
                )
                session_worker = None # Ensure it's None if totally failed


def process_image(image_path_str: str, output_path_str: str) -> tuple:
    """
    Removes the background from a single image and saves it.
    Uses the globally defined session_worker for the current process.
    """
    global session_worker
    if session_worker is None:
        return (
            image_path_str,
            False,
            "Session not initialized in worker.",
        )

    image_path = Path(image_path_str)
    output_path = Path(output_path_str)

    try:
        with open(image_path, "rb") as i:
            input_bytes = i.read()
        
        # Perform background removal
        output_bytes = remove(input_bytes, session=session_worker)

        # Ensure output directory exists (though created by main)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as o:
            o.write(output_bytes)
        return image_path_str, True, None
    except Exception as e:
        return image_path_str, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Remove background from images in a folder or a single image file. \
                     Optimized for Apple Silicon using CPU and GPU/ANE via CoreML."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input image file or folder containing images.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_nobg",
        help="Suffix to append to the output folder name or file name (default: _nobg).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="isnet-general-use", # Fast and good quality
        # Other options: "u2net", "u2netp", "silueta", "isnet-anime"
        help="Name of the rembg model to use (default: isnet-general-use). "
             "Models like u2net, u2netp, silueta, isnet-general-use, isnet-anime are available.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None, # os.cpu_count() will be used by ProcessPoolExecutor
        help="Maximum number of worker processes to use (default: all available CPU cores).",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force using CPUExecutionProvider only, even on Apple Silicon.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve() # Get absolute path
    model_name = args.model

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        return

    # Determine ONNX execution providers
    providers = ["CPUExecutionProvider"] # Default
    if platform.system() == "Darwin" and platform.machine() == "arm64" and not args.force_cpu:
        print("Apple Silicon (arm64) detected. Attempting to use CoreMLExecutionProvider.")
        # Order matters: CoreML first, then CPU as fallback
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    elif args.force_cpu:
        print("Forcing CPUExecutionProvider.")
    else:
        print(f"Running on {platform.system()} {platform.machine()}. Using CPUExecutionProvider.")
    
    # Test session creation in main thread to catch immediate provider issues
    try:
        print(f"Attempting to initialize a test session with model '{model_name}' and providers: {providers}")
        # This also triggers model download if not already cached
        _ = new_session(model_name=model_name, providers=providers)
        print(f"Test session for model '{model_name}' initialized successfully with providers: {providers}.")
        print(f"Note: The first run will download the model '{model_name}' if not cached (~20-170MB depending on model).")
    except Exception as e:
        print(f"Error initializing test session with providers {providers}: {e}")
        if "CoreMLExecutionProvider" in providers:
            print("Falling back to CPUExecutionProvider for all operations.")
            providers = ["CPUExecutionProvider"]
            try:
                _ = new_session(model_name=model_name, providers=providers)
                print(f"Test session for model '{model_name}' initialized successfully with CPUExecutionProvider.")
            except Exception as e_cpu:
                print(f"CRITICAL: Failed to initialize session even with CPUExecutionProvider: {e_cpu}")
                print("Please check your onnxruntime and rembg installation.")
                return
        else:
            print("CRITICAL: Failed to initialize session with CPUExecutionProvider.")
            print("Please check your onnxruntime and rembg installation.")
            return


    image_files_to_process = []
    output_base_dir = None

    if input_path.is_file():
        if input_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            image_files_to_process.append(input_path)
            # For a single file, create the output folder in its parent directory
            output_base_dir = input_path.parent / (input_path.stem + args.output_suffix + "_output")
        else:
            print(f"Error: '{input_path}' is not a supported image file (png, jpg, jpeg, webp).")
            return
    elif input_path.is_dir():
        print(f"Processing images in folder: {input_path}")
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            image_files_to_process.extend(list(input_path.glob(ext)))
        
        if not image_files_to_process:
            print(f"No image files found in '{input_path}'.")
            return
        # Create output folder next to the input folder
        output_base_dir = input_path.parent / (input_path.name + args.output_suffix)
    else:
        print(f"Error: Input path '{input_path}' is neither a file nor a directory.")
        return

    if not image_files_to_process:
        print("No images to process.")
        return

    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processed images will be saved in: {output_base_dir}")

    processed_count = 0
    error_count = 0
    errors_list = []

    # Using ProcessPoolExecutor to utilize multiple CPU cores
    # The initializer will create one rembg session per worker process
    num_workers = args.max_workers if args.max_workers else os.cpu_count()
    print(f"Using up to {num_workers} worker processes.")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(model_name, providers), # Pass model_name and providers to initializer
    ) as executor:
        futures = []
        for img_path in image_files_to_process:
            # Output file will always be PNG to support transparency
            output_file_name = img_path.stem + ".png" 
            output_file_path = output_base_dir / output_file_name
            futures.append(
                executor.submit(
                    process_image, str(img_path), str(output_file_path)
                )
            )

        for future in tqdm(
            as_completed(futures),
            total=len(image_files_to_process),
            desc="Processing images",
        ):
            img_path_str, success, error_msg = future.result()
            if success:
                processed_count += 1
            else:
                error_count += 1
                errors_list.append(f"Error processing {img_path_str}: {error_msg}")
                # print(f"Failed to process {img_path_str}: {error_msg}")


    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count} images.")
    print(f"Failed to process: {error_count} images.")
    if errors_list:
        print("\nErrors encountered:")
        for err in errors_list:
            print(f"- {err}")
    print(f"Output directory: {output_base_dir}")


if __name__ == "__main__":
    # This is important for multiprocessing on some platforms (like Windows)
    # and good practice.
    # import multiprocessing
    # multiprocessing.freeze_support() # Not strictly needed on macOS/Linux for this script
    main()
