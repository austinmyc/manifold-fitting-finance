"""
Step 1: Download embeddings from API and cache them.

This script downloads embeddings for both training and test data,
saving them for reuse without requiring API calls again.

Usage:
    python download_embeddings.py --split train
    python download_embeddings.py --split test
    python download_embeddings.py --split both
"""

import numpy as np
import argparse
import logging
import time
from pathlib import Path
from datasets import load_from_disk
from utils.zhipu_embedding import get_zhipu_embedding
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_embeddings.log'),
        logging.StreamHandler()
    ]
)


def get_embeddings_in_batches(sentences, batch_size=20):
    """Get embeddings for sentences in batches with rate limiting"""
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    all_embeddings = []

    for batch_idx in tqdm(range(num_batches), desc="Getting embeddings"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(sentences))
        batch_sentences = sentences[start_idx:end_idx]

        try:
            response = get_zhipu_embedding(model="embedding-3", input=batch_sentences)
            batch_embeddings = np.array([item.embedding for item in response.data])
            all_embeddings.append(batch_embeddings)

            # Rate limiting
            if batch_idx < num_batches - 1:
                time.sleep(0.5)

        except Exception as e:
            logging.error(f"Error processing batch {batch_idx+1}: {e}")
            raise

    return np.vstack(all_embeddings)


def download_train_embeddings(data_dir="data/finsent", output_file="data/train_embeddings_raw.npz"):
    """Download and cache training embeddings"""
    logging.info("="*70)
    logging.info("Downloading Training Embeddings")
    logging.info("="*70)

    # Check if already exists
    if Path(output_file).exists():
        logging.warning(f"{output_file} already exists. Skipping download.")
        logging.info("Use --force to re-download.")
        return

    # Load training data
    logging.info(f"Loading training data from {data_dir}...")
    dataset = load_from_disk(data_dir)
    train_data = dataset['train']

    # Extract sentences and labels
    train_sentences = [train_data[i]['text'] for i in range(len(train_data))]
    train_labels = np.array([train_data[i]['label'] for i in range(len(train_data))])

    logging.info(f"Total training samples: {len(train_sentences)}")

    # Download embeddings
    logging.info("Downloading embeddings from API...")
    train_embeddings = get_embeddings_in_batches(train_sentences, batch_size=20)

    logging.info(f"Embeddings shape: {train_embeddings.shape}")

    # Save to disk
    logging.info(f"Saving to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file,
             embeddings=train_embeddings,
             labels=train_labels,
             sentences=train_sentences)

    logging.info(f"✓ Saved training embeddings to {output_file}")
    logging.info(f"  - embeddings: {train_embeddings.shape}")
    logging.info(f"  - labels: {train_labels.shape}")
    logging.info(f"  - sentences: {len(train_sentences)}")


def download_test_embeddings(data_dir="data/finsent", output_file="data/test_embeddings_raw.npz"):
    """Download and cache test embeddings"""
    logging.info("="*70)
    logging.info("Downloading Test Embeddings")
    logging.info("="*70)

    # Check if already exists
    if Path(output_file).exists():
        logging.warning(f"{output_file} already exists. Skipping download.")
        logging.info("Use --force to re-download.")
        return

    # Load test data
    logging.info(f"Loading test data from {data_dir}...")
    dataset = load_from_disk(data_dir)
    test_data = dataset['test']

    # Extract sentences and labels
    test_sentences = [test_data[i]['text'] for i in range(len(test_data))]
    test_labels = np.array([test_data[i]['label'] for i in range(len(test_data))])

    logging.info(f"Total test samples: {len(test_sentences)}")

    # Download embeddings
    logging.info("Downloading embeddings from API...")
    test_embeddings = get_embeddings_in_batches(test_sentences, batch_size=20)

    logging.info(f"Embeddings shape: {test_embeddings.shape}")

    # Save to disk
    logging.info(f"Saving to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file,
             embeddings=test_embeddings,
             labels=test_labels,
             sentences=test_sentences)

    logging.info(f"✓ Saved test embeddings to {output_file}")
    logging.info(f"  - embeddings: {test_embeddings.shape}")
    logging.info(f"  - labels: {test_labels.shape}")
    logging.info(f"  - sentences: {len(test_sentences)}")


def main():
    parser = argparse.ArgumentParser(description='Download and cache embeddings from API')
    parser.add_argument('--split', type=str, default='both',
                       choices=['train', 'test', 'both'],
                       help='Which split to download (default: both)')
    parser.add_argument('--data-dir', type=str, default='data/finsent',
                       help='Path to dataset directory')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')

    args = parser.parse_args()

    # Remove existing files if force flag is set
    if args.force:
        for f in ['data/train_embeddings_raw.npz', 'data/test_embeddings_raw.npz']:
            if Path(f).exists():
                logging.info(f"Removing existing file: {f}")
                Path(f).unlink()

    # Download embeddings
    if args.split in ['train', 'both']:
        download_train_embeddings(data_dir=args.data_dir)

    if args.split in ['test', 'both']:
        download_test_embeddings(data_dir=args.data_dir)

    logging.info("\n" + "="*70)
    logging.info("✓ Download complete!")
    logging.info("="*70)
    logging.info("\nNext steps:")
    logging.info("  1. Run: python transform_embeddings.py")
    logging.info("  2. Run: python train_mlp.py")


if __name__ == "__main__":
    main()
