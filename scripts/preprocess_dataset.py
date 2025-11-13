"""
Dataset Preprocessing Script
Processes LFW-FER and DeeperForensics-1.0 datasets
"""

import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime
import sys

# Add project src to path (project_root/src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from preprocessing.pipeline import PreprocessingPipeline, PreprocessingOutput
from utils.logger import setup_logger


def load_lfw_fer_dataset(config: dict) -> list:
    """
    Load LFW-FER dataset

    Returns:
        List of (image_path, image_id, dataset_name, label) tuples
    """
    dataset_path = Path(config['datasets']['lfw_fer']['path'])
    image_ext = config['datasets']['lfw_fer']['image_extension']

    # Find all images
    image_paths = list(dataset_path.rglob(f"*{image_ext}"))

    dataset_items = []
    for img_path in image_paths:
        image_id = img_path.stem
        dataset_items.append((img_path, image_id, 'lfw_fer', 'real'))

    return dataset_items


def load_deeper_forensics_dataset(config: dict) -> list:
    """
    Load DeeperForensics-1.0 dataset

    Returns:
        List of (image_path, image_id, dataset_name, label) tuples
    """
    dataset_path = Path(config['datasets']['deeper_forensics']['path'])
    real_folder = config['datasets']['deeper_forensics']['real_folder']
    fake_folder = config['datasets']['deeper_forensics']['fake_folder']
    image_ext = config['datasets']['deeper_forensics'].get('image_extension', None)

    # Video handling config (with safe defaults)
    video_extensions = config['datasets']['deeper_forensics'].get(
        'video_extensions', ['.mp4', '.avi', '.mov', '.mkv']
    )
    frame_stride = config['datasets']['deeper_forensics'].get('frame_stride', 30)  # every 30th frame
    max_frames_per_video = config['datasets']['deeper_forensics'].get('max_frames_per_video', 10)

    dataset_items = []

    def collect_from_folder(base_path: Path, label: str):
        # 1) Pre-extracted images
        if image_ext is not None:
            image_files = list(base_path.rglob(f"*{image_ext}"))
            for img_path in image_files:
                image_id = img_path.stem
                dataset_items.append((img_path, image_id, 'deeper_forensics', label))

        # 2) Video files → sample frames
        video_files = [p for p in base_path.rglob('*') if p.suffix.lower() in video_extensions]
        for vid_path in video_files:
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                continue
            frame_count = 0
            saved_count = 0
            while True:
                ret = cap.grab()  # grab first for efficiency
                if not ret:
                    break
                if frame_count % frame_stride == 0:
                    ret2, frame_bgr = cap.retrieve()
                    if not ret2:
                        break
                    # We pass raw frame (BGR); the processing loop will convert to RGB
                    image_id = f"{vid_path.stem}_f{frame_count}"
                    dataset_items.append((frame_bgr, image_id, 'deeper_forensics', label))
                    saved_count += 1
                    if saved_count >= max_frames_per_video:
                        break
                frame_count += 1
            cap.release()

    # Load real
    real_path = dataset_path / real_folder
    if real_path.exists():
        collect_from_folder(real_path, 'real')

    # Load fake
    fake_path = dataset_path / fake_folder
    if fake_path.exists():
        collect_from_folder(fake_path, 'fake')

    return dataset_items


def process_dataset(pipeline: PreprocessingPipeline,
                    dataset_items: list,
                    output_dir: Path,
                    logger,
                    save_visualizations: bool = False) -> pd.DataFrame:
    """
    Process entire dataset through pipeline

    Args:
        pipeline: PreprocessingPipeline instance
        dataset_items: List of dataset items
        output_dir: Output directory
        logger: Logger instance
        save_visualizations: Whether to save visualization images

    Returns:
        DataFrame with processing results
    """
    results = []
    failed_count = 0

    # Create visualization directory if needed
    if save_visualizations:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(dataset_items)} images...")

    for img_path, image_id, dataset_name, label in tqdm(dataset_items, desc="Processing"):
        try:
            # Load image or accept preloaded frame
            if isinstance(img_path, np.ndarray):
                image = img_path
            else:
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    failed_count += 1
                    continue

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process through pipeline
            output = pipeline.process_image(image, image_id, dataset_name, label)

            if output is None:
                logger.warning(f"Failed to process: {image_id}")
                failed_count += 1
                results.append({
                    'image_id': image_id,
                    'dataset': dataset_name,
                    'label': label,
                    'processed': False,
                    'reason': 'detection_failed'
                })
                continue

            # Save output
            saved_paths = pipeline.save_output(output, output_dir)

            # Save visualization if requested
            if save_visualizations and output.is_valid:
                vis_image = pipeline.visualize_pipeline(image, output)
                vis_path = vis_dir / f"{dataset_name}_{label}_{image_id}_vis.jpg"
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

            # Record result
            result_dict = {
                'image_id': image_id,
                'dataset': dataset_name,
                'label': label,
                'processed': True,
                'is_valid': output.is_valid,
                'quality_score': output.quality_score,
                'detection_confidence': output.detection_confidence,
                'face_path': str(saved_paths['face'].relative_to(output_dir)),
                'landmarks_path': str(saved_paths['landmarks'].relative_to(output_dir)),
                'metadata_path': str(saved_paths['metadata'].relative_to(output_dir))
            }

            # Add quality metrics
            for metric_name, metric_value in output.quality_metrics.items():
                result_dict[f'quality_{metric_name}'] = metric_value

            results.append(result_dict)

        except Exception as e:
            logger.error(f"Error processing {image_id}: {str(e)}")
            failed_count += 1
            results.append({
                'image_id': image_id,
                'dataset': dataset_name,
                'label': label,
                'processed': False,
                'reason': str(e)
            })

    logger.info(f"Processing complete. Failed: {failed_count}/{len(dataset_items)}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def create_data_splits(results_df: pd.DataFrame,
                       output_dir: Path,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       random_seed: int = 42) -> dict:
    """
    Create train/val/test splits

    Args:
        results_df: Results DataFrame
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with split DataFrames
    """
    # Filter only successfully processed images
    valid_df = results_df[results_df['processed'] == True].copy()

    np.random.seed(random_seed)

    splits = {}

    # Process each dataset and label separately to ensure balance
    for dataset in valid_df['dataset'].unique():
        for label in valid_df['label'].unique():
            subset = valid_df[(valid_df['dataset'] == dataset) &
                              (valid_df['label'] == label)]

            if len(subset) == 0:
                continue

            # Shuffle
            subset = subset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

            # Calculate split indices
            n = len(subset)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            # Split
            train_subset = subset[:train_end]
            val_subset = subset[train_end:val_end]
            test_subset = subset[val_end:]

            # Append to splits
            for split_name, split_data in [('train', train_subset),
                                           ('val', val_subset),
                                           ('test', test_subset)]:
                if split_name not in splits:
                    splits[split_name] = []
                splits[split_name].append(split_data)

    # Concatenate and save
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    final_splits = {}
    for split_name, split_list in splits.items():
        split_df = pd.concat(split_list, ignore_index=True)
        split_df = split_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Save
        split_path = splits_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)

        final_splits[split_name] = split_df

        print(f"{split_name.capitalize()} set: {len(split_df)} images")
        print(f"  Real: {len(split_df[split_df['label'] == 'real'])}")
        print(f"  Fake: {len(split_df[split_df['label'] == 'fake'])}")

    return final_splits


def main():
    parser = argparse.ArgumentParser(description='Preprocess deepfake detection datasets')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--datasets', nargs='+',
                        choices=['lfw_fer', 'deeper_forensics', 'gen_ai', 'all'],
                        default=['all'],
                        help='Datasets to process')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for testing)')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"preprocessing_{timestamp}.log"

    logger = setup_logger('preprocessing', log_file)
    logger.info(f"Starting preprocessing with config: {args.config}")

    # Initialize pipeline
    pipeline = PreprocessingPipeline(config)

    # Load datasets
    dataset_items = []

    if 'all' in args.datasets or 'lfw_fer' in args.datasets:
        logger.info("Loading LFW-FER dataset...")
        lfw_items = load_lfw_fer_dataset(config)
        dataset_items.extend(lfw_items)
        logger.info(f"Loaded {len(lfw_items)} images from LFW-FER")

    if 'all' in args.datasets or 'deeper_forensics' in args.datasets:
        logger.info("Loading DeeperForensics-1.0 dataset...")
        df_items = load_deeper_forensics_dataset(config)
        dataset_items.extend(df_items)
        logger.info(f"Loaded {len(df_items)} images from DeeperForensics-1.0")

    if 'all' in args.datasets or 'gen_ai' in args.datasets:
        if 'gen_ai' in config['datasets']:
            logger.info("Loading GenAI dataset...")
            # (중요) 기존 함수를 재사용하되, gen_ai 설정을 전달
            genai_config = config['datasets']['gen_ai']
            genai_items = load_deeper_forensics_dataset(genai_config)
            dataset_items.extend(genai_items)
            logger.info(f"Loaded {len(genai_items)} items from GenAI")
        else:
            logger.warning("'gen_ai' dataset selected but not defined in config file.")

    # Limit if specified
    if args.max_images is not None:
        dataset_items = dataset_items[:args.max_images]
        logger.info(f"Limited to {args.max_images} images for testing")

    # Setup output directory
    output_dir = Path(config['output']['base_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process dataset
    logger.info(f"Processing {len(dataset_items)} total images...")
    results_df = process_dataset(
        pipeline,
        dataset_items,
        output_dir,
        logger,
        save_visualizations=args.visualize
    )

    # Save results
    results_path = output_dir / f'preprocessing_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to: {results_path}")

    # Compute and save statistics
    successful_outputs = [
        pipeline.load_output(output_dir,
                             f"{row['dataset']}_{row['label']}_{row['image_id']}")
        for _, row in results_df[results_df['processed'] == True].iterrows()
    ]

    stats = pipeline.get_statistics(successful_outputs)
    stats_path = output_dir / f'statistics_{timestamp}.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to: {stats_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total images: {len(dataset_items)}")
    print(f"Successfully processed: {len(results_df[results_df['processed'] == True])}")
    print(f"Failed: {len(results_df[results_df['processed'] == False])}")
    print(f"Valid quality: {len(results_df[results_df['is_valid'] == True])}")
    print(f"Average quality score: {stats['avg_quality_score']:.3f}")
    print(f"Average detection confidence: {stats['avg_detection_confidence']:.3f}")
    print("=" * 50)

    # Create data splits
    logger.info("Creating train/val/test splits...")
    splits = create_data_splits(results_df, output_dir)

    logger.info("Preprocessing complete!")


if __name__ == '__main__':
    main()