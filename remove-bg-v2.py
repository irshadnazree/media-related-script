# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "rembg[cpu]",
#     "tqdm",
#     "Pillow",
# ]
# ///
from __future__ import annotations

import argparse
import os
import platform
import shlex
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlparse

from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# --- Global session for worker processes ---
# This avoids re-initializing the model for every image in a single process.
# Each process in the pool will have its own session_worker.
session_worker = None
remove_worker = None


def load_rembg():
    """Import rembg lazily because its numpy/numba stack can take a moment to load."""
    try:
        from rembg import new_session, remove
    except ModuleNotFoundError as e:
        if e.name == "onnxruntime":
            raise RuntimeError(
                "rembg could not import onnxruntime. Run this script with uv again "
                "so the inline dependency 'rembg[cpu]' is installed."
            ) from e
        raise

    return new_session, remove


def normalize_pasted_path(raw_path: str) -> Path:
    """Handle drag-and-drop terminal paths, quoted paths, and file:// URLs."""
    path_text = raw_path.strip()
    if not path_text:
        raise ValueError("No input path provided.")

    if path_text.startswith("file://"):
        parsed = urlparse(path_text)
        path_text = unquote(parsed.path)
    else:
        try:
            parts = shlex.split(path_text)
        except ValueError:
            parts = [path_text.strip("'\"")]
        if parts:
            first_path = Path(parts[0]).expanduser()
            path_text = parts[0] if first_path.exists() else " ".join(parts)

    return Path(path_text).expanduser().resolve()


def prompt_for_input_path() -> Path:
    print("Paste or drag an image/folder path here, then press Return:")
    return normalize_pasted_path(input("> "))


def collect_image_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def output_base_for(input_path: Path, output_suffix: str) -> Path:
    if input_path.is_file():
        return input_path.parent / f"{input_path.stem}{output_suffix}_output"
    return input_path.parent / f"{input_path.name}{output_suffix}"


def build_jobs(
    image_files: list[Path],
    output_base_dir: Path,
    input_path: Path,
    overwrite: bool,
) -> tuple[list[tuple[Path, Path]], int]:
    jobs = []
    skipped_count = 0

    for img_path in image_files:
        if input_path.is_dir():
            relative_path = img_path.relative_to(input_path)
            output_file_path = output_base_dir / relative_path.with_suffix(".png")
        else:
            output_file_path = output_base_dir / f"{img_path.stem}.png"

        if output_file_path.exists() and not overwrite:
            skipped_count += 1
            continue

        jobs.append((img_path, output_file_path))

    return jobs, skipped_count


def choose_providers(force_cpu: bool) -> list[str]:
    if force_cpu:
        print("Forcing CPUExecutionProvider.")
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort

        available_providers = set(ort.get_available_providers())
    except Exception:
        available_providers = set()

    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and "CoreMLExecutionProvider" in available_providers
    ):
        print("Apple Silicon CoreML provider detected.")
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("Apple Silicon detected, but CoreML provider is not available. Using CPUExecutionProvider.")
    else:
        print(f"Running on {platform.system()} {platform.machine()}. Using CPUExecutionProvider.")

    return ["CPUExecutionProvider"]


def choose_worker_count(requested_workers: int | None, image_count: int, providers: list[str]) -> int:
    if image_count <= 1:
        return 1
    if requested_workers is not None:
        return max(1, min(requested_workers, image_count))
    if "CoreMLExecutionProvider" in providers:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count, image_count))


def init_worker(model_name_init, providers_init):
    """Initializer for each worker process in the pool."""
    global session_worker, remove_worker
    try:
        new_session, remove_worker = load_rembg()
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
                new_session, remove_worker = load_rembg()
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
    global session_worker, remove_worker
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
        if remove_worker is None:
            _, remove_worker = load_rembg()
        output_bytes = remove_worker(input_bytes, session=session_worker)

        # Ensure output directory exists (though created by main)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as o:
            o.write(output_bytes)
        return image_path_str, True, None
    except Exception as e:
        return image_path_str, False, str(e)


def process_jobs_in_current_process(
    jobs: list[tuple[Path, Path]],
    model_name: str,
    providers: list[str],
) -> tuple[int, int, list[str]]:
    new_session, remove_func = load_rembg()
    session = new_session(model_name=model_name, providers=providers)
    processed_count = 0
    error_count = 0
    errors_list = []

    for img_path, output_file_path in tqdm(jobs, total=len(jobs), desc="Processing images"):
        try:
            input_bytes = img_path.read_bytes()
            output_bytes = remove_func(input_bytes, session=session)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            output_file_path.write_bytes(output_bytes)
            processed_count += 1
        except Exception as e:
            error_count += 1
            errors_list.append(f"Error processing {img_path}: {e}")

    return processed_count, error_count, errors_list


def process_jobs_in_worker_pool(
    jobs: list[tuple[Path, Path]],
    model_name: str,
    providers: list[str],
    num_workers: int,
) -> tuple[int, int, list[str]]:
    processed_count = 0
    error_count = 0
    errors_list = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(model_name, providers),
    ) as executor:
        futures = [
            executor.submit(process_image, str(img_path), str(output_file_path))
            for img_path, output_file_path in jobs
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(jobs),
            desc="Processing images",
        ):
            img_path_str, success, error_msg = future.result()
            if success:
                processed_count += 1
            else:
                error_count += 1
                errors_list.append(f"Error processing {img_path_str}: {error_msg}")

    return processed_count, error_count, errors_list


def main():
    parser = argparse.ArgumentParser(
        description="Remove background from images in a folder or a single image file. \
                     Optimized for Apple Silicon using CPU and GPU/ANE via CoreML."
    )
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        help="Path to the input image file or folder containing images. If omitted, you will be prompted to paste or drag a path.",
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
        default=None,
        help="Maximum number of worker processes to use (default: 1 for CoreML, up to 4 for CPU batches).",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force using CPUExecutionProvider only, even on Apple Silicon.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images inside nested folders when the input path is a folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even when the output PNG already exists.",
    )

    args = parser.parse_args()

    try:
        input_path = (
            normalize_pasted_path(args.input_path)
            if args.input_path
            else prompt_for_input_path()
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    model_name = args.model

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        return

    if not input_path.is_file() and not input_path.is_dir():
        print(f"Error: Input path '{input_path}' is neither a file nor a directory.")
        return

    if input_path.is_dir():
        print(f"Processing images in folder: {input_path}")

    image_files_to_process = collect_image_files(input_path, args.recursive)
    if not image_files_to_process:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"No supported image files found. Supported extensions: {supported}.")
        return

    output_base_dir = output_base_for(input_path, args.output_suffix)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processed images will be saved in: {output_base_dir}")

    jobs, skipped_count = build_jobs(
        image_files_to_process,
        output_base_dir,
        input_path,
        args.overwrite,
    )
    if skipped_count:
        print(f"Skipped existing outputs: {skipped_count} files. Use --overwrite to reprocess them.")
    if not jobs:
        print("Nothing to process.")
        print(f"Output directory: {output_base_dir}")
        return

    providers = choose_providers(args.force_cpu)
    num_workers = choose_worker_count(args.max_workers, len(jobs), providers)

    print(
        "Loading rembg and its image-processing dependencies. This can take a moment on first run.",
        flush=True,
    )
    print(f"Model: {model_name}")
    print(f"Execution providers: {providers}")
    print(f"Images queued: {len(jobs)}")

    try:
        if num_workers == 1:
            print("Using one in-process worker to avoid extra model startup overhead.")
            processed_count, error_count, errors_list = process_jobs_in_current_process(
                jobs,
                model_name,
                providers,
            )
        else:
            print(f"Using {num_workers} worker processes.")
            processed_count, error_count, errors_list = process_jobs_in_worker_pool(
                jobs,
                model_name,
                providers,
                num_workers,
            )
    except Exception as e:
        if "CoreMLExecutionProvider" not in providers:
            print(f"CRITICAL: Failed to process images: {e}")
            print("Please check your onnxruntime and rembg installation.")
            return

        print(f"CoreML processing failed: {e}")
        print("Falling back to CPUExecutionProvider.")
        providers = ["CPUExecutionProvider"]
        processed_count, error_count, errors_list = process_jobs_in_current_process(
            jobs,
            model_name,
            providers,
        )


    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count} images.")
    if skipped_count:
        print(f"Skipped existing outputs: {skipped_count} images.")
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
