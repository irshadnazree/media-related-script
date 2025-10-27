#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pillow>=10.0.0",
#     "tqdm>=4.66.0",
# ]
# ///
"""
Image Processing Script
Supports batch processing with compression, conversion, and renaming.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
import multiprocessing

from PIL import Image
from tqdm import tqdm


class ImageFormat(Enum):
    """Supported image formats."""

    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    TIFF = "tiff"
    BMP = "bmp"
    ICO = "ico"


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".ico",
}

FORMAT_MAP = {
    "1": ImageFormat.JPEG,
    "2": ImageFormat.PNG,
    "3": ImageFormat.WEBP,
    "4": ImageFormat.GIF,
    "5": ImageFormat.TIFF,
    "6": ImageFormat.BMP,
    "7": ImageFormat.ICO,
}


@dataclass
class ProcessingConfig:
    """Configuration for image processing operations."""

    compress: bool = False
    quality: int = 85
    convert: bool = False
    target_format: Optional[ImageFormat] = None
    rename: bool = False
    base_name: str = ""
    keep_resolution: bool = True
    max_height: int = 1600


class ImageProcessor:
    """Handles image processing operations."""

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process_image(
        self, input_path: Path, output_path: Path
    ) -> Tuple[bool, Optional[str]]:
        """Process a single image with configured operations."""
        try:
            with Image.open(input_path) as img:
                img = self._apply_transformations(img)
                self._save_image(img, output_path, img.format)
            return True, None

        except FileNotFoundError:
            return False, f"File not found: {input_path}"
        except Image.UnidentifiedImageError:
            return False, f"Cannot identify image: {input_path}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def _apply_transformations(self, img: Image.Image) -> Image.Image:
        """Apply resize, conversion, and other transformations."""
        # Resize if needed
        if (
            self.config.compress
            and not self.config.keep_resolution
            and img.height > self.config.max_height
        ):
            img = self._resize_image(img)

        # Convert format if needed
        if self.config.convert and self.config.target_format:
            img = self._convert_format(img, self.config.target_format)

        return img

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        aspect_ratio = img.width / img.height
        new_height = self.config.max_height
        new_width = int(new_height * aspect_ratio)

        print(f"Resizing from {img.width}x{img.height} to {new_width}x{new_height}")
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _convert_format(
        self, img: Image.Image, target_format: ImageFormat
    ) -> Image.Image:
        """Convert image to target format with proper mode handling."""
        if target_format == ImageFormat.JPEG:
            return self._convert_to_jpeg(img)
        elif target_format == ImageFormat.ICO:
            return self._convert_to_ico(img)
        elif target_format == ImageFormat.PNG:
            return self._ensure_mode(img, ["RGB", "RGBA"])
        else:
            return self._ensure_mode(img, ["RGB", "RGBA"])

    def _convert_to_jpeg(self, img: Image.Image) -> Image.Image:
        """Convert image to JPEG, handling transparency."""
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, (0, 0), img.split()[-1])
            return background
        return img.convert("RGB") if img.mode != "RGB" else img

    def _convert_to_ico(self, img: Image.Image) -> Image.Image:
        """Convert image to ICO format."""
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        if img.width != img.height:
            print(
                f"Warning: ICO works best with square images. "
                f"Current: {img.width}x{img.height}"
            )
        return img

    def _ensure_mode(self, img: Image.Image, allowed_modes: List[str]) -> Image.Image:
        """Ensure image is in one of the allowed modes."""
        if img.mode not in allowed_modes:
            target_mode = "RGBA" if "A" in img.mode else "RGB"
            return img.convert(target_mode)
        return img

    def _save_image(
        self, img: Image.Image, output_path: Path, original_format: str
    ) -> None:
        """Save image with appropriate options."""
        save_options = {}

        if self.config.compress:
            target_format = (
                self.config.target_format.value
                if self.config.convert
                else original_format
            )

            if target_format.lower() in ["jpeg", "webp"]:
                save_options["quality"] = self.config.quality
            elif target_format.lower() == "png":
                save_options["optimize"] = True

        save_format = (
            self.config.target_format.value.upper()
            if self.config.convert
            else original_format
        )

        img.save(output_path, format=save_format, **save_options)


class BatchProcessor:
    """Handles batch processing of multiple images."""

    def __init__(self, processor: ImageProcessor, output_dir: Path):
        self.processor = processor
        self.output_dir = output_dir

    def process_batch(
        self,
        image_files: List[Path],
        config: ProcessingConfig,
    ) -> Tuple[int, int]:
        """Process multiple images concurrently."""
        max_workers = min(multiprocessing.cpu_count() * 2, len(image_files), 32)
        print(f"\nProcessing {len(image_files)} images using {max_workers} workers...")

        args_list = [
            (i, img_path, config, len(image_files))
            for i, img_path in enumerate(image_files)
        ]

        successful = 0
        failed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._process_single, args): args[1]
                for args in args_list
            }

            with tqdm(total=len(image_files), desc="Progress", unit="img") as pbar:
                for future in concurrent.futures.as_completed(future_to_path):
                    success, error = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        if error:
                            tqdm.write(f"✗ {error}")
                    pbar.update(1)

        return successful, failed

    def _process_single(
        self, args: Tuple[int, Path, ProcessingConfig, int]
    ) -> Tuple[bool, Optional[str]]:
        """Process a single image with index and config."""
        index, input_path, config, total = args

        try:
            output_path = self._generate_output_path(input_path, index, total, config)
            return self.processor.process_image(input_path, output_path)
        except Exception as e:
            return False, f"Error processing {input_path.name}: {str(e)}"

    def _generate_output_path(
        self,
        input_path: Path,
        index: int,
        total: int,
        config: ProcessingConfig,
    ) -> Path:
        """Generate output path based on configuration."""
        original_stem = input_path.stem

        # Determine new name
        if config.rename and config.base_name:
            name_part = (
                f"{config.base_name}-{index + 1}" if total > 1 else config.base_name
            )
        else:
            name_part = original_stem

        # Determine new extension
        if config.convert and config.target_format:
            extension = f".{config.target_format.value}"
        else:
            extension = input_path.suffix.lower()

        # Add suffix for single file if not renamed/converted
        if total == 1 and not config.rename and not config.convert:
            name_part = f"{name_part}_processed"

        return self.output_dir / f"{name_part}{extension}"


class InputHandler:
    """Handles user input and validation."""

    @staticmethod
    def get_bool_input(prompt: str) -> bool:
        """Get yes/no input from user."""
        while True:
            choice = input(f"{prompt} (y/n): ").strip().lower()
            if choice in ["y", "yes"]:
                return True
            elif choice in ["n", "no"]:
                return False
            print("Invalid input. Please enter 'y' or 'n'.")

    @staticmethod
    def get_path_input() -> Path:
        """Get and validate path input from user."""
        while True:
            raw_path = input("Enter path to image file or directory: ").strip()

            path = InputHandler._clean_path(raw_path)

            if path.exists():
                return path

            print("Error: Path does not exist. Please try again.")
            print('Tip: Wrap paths with spaces in quotes: "/path/"')

    @staticmethod
    def _clean_path(raw_path: str) -> Path:
        """Clean and normalize path string."""
        if (raw_path.startswith('"') and raw_path.endswith('"')) or (
            raw_path.startswith("'") and raw_path.endswith("'")
        ):
            raw_path = raw_path[1:-1]

        raw_path = raw_path.replace("\\ ", " ")
        return Path(raw_path).expanduser().resolve()

    @staticmethod
    def get_conversion_config() -> Tuple[bool, Optional[ImageFormat]]:
        """Get format conversion configuration."""
        print("\nConvert image format?")
        print("Format: 'y [format_number]' or 'n'")
        print("Formats: 1=JPEG 2=PNG 3=WEBP 4=GIF 5=TIFF 6=BMP 7=ICO")

        while True:
            parts = input("Enter settings: ").strip().lower().split()

            if not parts:
                print("Invalid input.")
                continue

            if parts[0] in ["n", "no"]:
                return False, None

            if parts[0] not in ["y", "yes"]:
                print("Invalid input. Start with 'y' or 'n'.")
                continue

            format_choice = parts[1] if len(parts) >= 2 else "1"
            target_format = FORMAT_MAP.get(format_choice, ImageFormat.JPEG)

            if format_choice not in FORMAT_MAP and len(parts) >= 2:
                print(f"Invalid format '{format_choice}'. Using JPEG.")

            return True, target_format

    @staticmethod
    def get_compression_config() -> Tuple[bool, int, bool]:
        """Get compression configuration."""
        print("\nCompress images?")
        print("Format: 'y [quality:1-100] [allow_downsize:y/n]' or 'n'")

        while True:
            parts = input("Enter settings: ").strip().lower().split()

            if not parts:
                print("Invalid input.")
                continue

            if parts[0] in ["n", "no"]:
                return False, 85, True

            if parts[0] not in ["y", "yes"]:
                print("Invalid input. Start with 'y' or 'n'.")
                continue

            quality = 85
            if len(parts) >= 2:
                try:
                    q = int(parts[1])
                    quality = max(1, min(100, q))
                    if not (1 <= q <= 100):
                        print("Quality clamped to 1-100 range.")
                except ValueError:
                    print(f"Invalid quality '{parts[1]}'. Using 85.")

            allow_downsize = True
            if len(parts) >= 3:
                if parts[2] in ["n", "no"]:
                    allow_downsize = False
                elif parts[2] not in ["y", "yes"]:
                    print(f"Invalid downsize option '{parts[2]}'. Using 'y'.")

            return True, quality, allow_downsize


def collect_image_files(path: Path) -> List[Path]:
    """Collect all image files from path (file or directory)."""
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        else:
            print(f"Unsupported file type: {path}")
            return []

    if path.is_dir():
        images = [
            f
            for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(images)

    return []


def setup_output_directory(input_path: Path) -> Path:
    """Create and return output directory path."""
    if input_path.is_file():
        output_dir = input_path.parent
        print(f"Output location: {output_dir}")
    else:
        output_dir = input_path / f"{input_path.name}_processed"
        output_dir.mkdir(exist_ok=True)
        print(f"Output location: {output_dir}")

    return output_dir


def main():
    """Main entry point for the image processor."""
    print("=== Image Processing Script ===\n")

    try:
        input_path = InputHandler.get_path_input()

        do_convert, target_format = InputHandler.get_conversion_config()
        do_compress, quality, allow_downsize = InputHandler.get_compression_config()

        do_rename = InputHandler.get_bool_input("Rename images?")
        base_name = ""
        if do_rename:
            base_name = input("Enter base name (e.g., 'holiday'): ").strip()

        config = ProcessingConfig(
            compress=do_compress,
            quality=quality,
            convert=do_convert,
            target_format=target_format,
            rename=do_rename,
            base_name=base_name,
            keep_resolution=not allow_downsize,
        )

        image_files = collect_image_files(input_path)
        if not image_files:
            print("No image files found to process.")
            return

        print(f"\nFound {len(image_files)} image(s)")

        output_dir = setup_output_directory(input_path)

        processor = ImageProcessor(config)
        batch_processor = BatchProcessor(processor, output_dir)
        successful, failed = batch_processor.process_batch(image_files, config)

        print("\n" + "=" * 50)
        print("Processing Complete")
        print("=" * 50)
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed:     {failed}")
        print(f"📁 Output:    {output_dir}")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
