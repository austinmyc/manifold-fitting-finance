import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from utils.zhipu_embedding import get_zhipu_embedding
from manfit.manfit_ours_gpu import manfit_ours_gpu_batched as manfit_ours
from manfit.tuning import quality_score
import logging
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_finsent.log'),
        logging.StreamHandler()
    ]
)


def main():
    logging.info("="*70)
    logging.info("Starting FinSent Embedding Processing Pipeline")
    logging.info("="*70)

    # Step 1: Load data
    logging.info("Step 1: Loading FinSent train data...")
    dataset = load_from_disk("data/finsent")
    train_data = dataset['train']
    total_samples = len(train_data)
    logging.info(f"Total train samples: {total_samples}")

    # Get all sentences
    all_sentences = [train_data[i]['text'] for i in range(total_samples)]
    logging.info(f"Extracted {len(all_sentences)} sentences")

    # Step 2: Process in batches
    batch_size = 20
    num_batches = (total_samples + batch_size - 1) // batch_size
    logging.info(f"\nStep 2: Processing {total_samples} sentences in batches of {batch_size}")
    logging.info(f"Total batches: {num_batches}")

    all_embeddings = []

    for batch_idx in tqdm(range(num_batches), desc="Getting embeddings"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        batch_sentences = all_sentences[start_idx:end_idx]

        logging.info(f"Batch {batch_idx+1}/{num_batches}: Processing sentences {start_idx} to {end_idx-1}")

        try:
            # Call ZhiPu API for this batch
            response = get_zhipu_embedding(model="embedding-3", input=batch_sentences)

            # Extract embeddings
            batch_embeddings = np.array([item.embedding for item in response.data])
            all_embeddings.append(batch_embeddings)

            logging.info(f"Batch {batch_idx+1}: Got embeddings of shape {batch_embeddings.shape}")

            # Rate limiting: small delay between batches
            if batch_idx < num_batches - 1:
                time.sleep(0.5)

        except Exception as e:
            logging.error(f"Error processing batch {batch_idx+1}: {e}")
            raise

    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    logging.info(f"\nCombined embeddings shape: {embeddings.shape}")
    logging.info(f"Expected shape: ({total_samples}, 2048)")

    # Verify dimensions
    assert embeddings.shape[1] == 2048, f"Unexpected embedding dimension: {embeddings.shape[1]}"
    logging.info("✓ Embedding dimensions verified")

    # Step 3: Tuning
    logging.info("\n" + "="*70)
    logging.info("Step 3: Splitting data for tuning...")
    # Use a subset for tuning to save time (max 1000 samples)
    tuning_size = min(1000, len(embeddings))
    tuning_embeddings = embeddings[:tuning_size]
    logging.info(f"Using {tuning_size} samples for hyperparameter tuning")

    # Split embeddings for tuning (80/20 split)
    train_emb, val_emb = train_test_split(tuning_embeddings, test_size=0.2, random_state=42)
    logging.info(f"Train embeddings: {train_emb.shape}")
    logging.info(f"Validation embeddings: {val_emb.shape}")

    logging.info("\nStep 4: Tuning sigma parameter using Bayesian optimization...")
    # Define search space for sigma
    space = [Real(0.01, 1.0, name='sig')]

    # Define objective function
    @use_named_args(space)
    def objective(sig):
        """Objective function to minimize"""
        # Transform validation embeddings using training embeddings as reference
        transformed_val = manfit_ours(sample=train_emb, sample_init=val_emb, sig=sig)

        # Compute quality score
        score = quality_score(transformed_val, intrinsic_dim=2, k=5)

        logging.info(f"  sig={sig:.4f}: quality_score={score:.6f}")

        return score

    # Run Bayesian optimization
    logging.info("Running optimization...")
    result = gp_minimize(
        objective,
        space,
        n_calls=15,           # Number of evaluations
        n_initial_points=5,   # Random exploration points
        random_state=42,
        verbose=False
    )

    best_sig = result.x[0]
    best_score = result.fun

    logging.info("\n" + "=" * 70)
    logging.info(f"Optimization complete!")
    logging.info(f"Best sigma: {best_sig:.4f}")
    logging.info(f"Best quality score: {best_score:.6f}")
    logging.info("=" * 70)

    # Step 5: Transform all embeddings
    logging.info("\n" + "="*70)
    logging.info(f"Step 5: Applying transformation to all {len(embeddings)} embeddings...")
    logging.info(f"Using best sigma: {best_sig:.4f}")

    # Apply transformation using best sigma
    transformed_embeddings = manfit_ours(
        sample=embeddings,
        sig=best_sig,
        sample_init=embeddings
    )

    logging.info(f"Transformed embeddings shape: {transformed_embeddings.shape}")

    # Show some statistics
    logging.info("\n" + "="*70)
    logging.info("Original vs Transformed comparison:")
    logging.info(f"Original - Mean: {np.mean(embeddings):.6f}, Std: {np.std(embeddings):.6f}")
    logging.info(f"Transformed - Mean: {np.mean(transformed_embeddings):.6f}, Std: {np.std(transformed_embeddings):.6f}")

    # Calculate the difference
    diff = transformed_embeddings - embeddings
    logging.info(f"\nDifference - Mean: {np.mean(diff):.6f}, Std: {np.std(diff):.6f}")
    logging.info(f"Max absolute change: {np.max(np.abs(diff)):.6f}")
    logging.info(f"Min absolute change: {np.min(np.abs(diff)):.6f}")

    # Save results
    logging.info("\n" + "="*70)
    logging.info("Step 6: Saving results...")

    output_file = 'finsent_embeddings.npz'
    np.savez(output_file,
             original=embeddings,
             transformed=transformed_embeddings,
             sentences=all_sentences,
             best_sigma=best_sig,
             best_score=best_score)

    logging.info(f"✓ Saved: {output_file}")
    logging.info(f"  - original: {embeddings.shape}")
    logging.info(f"  - transformed: {transformed_embeddings.shape}")
    logging.info(f"  - sentences: {len(all_sentences)} texts")
    logging.info(f"  - best_sigma: {best_sig:.4f}")
    logging.info(f"  - best_score: {best_score:.6f}")
    logging.info(f"\nTo load: data = np.load('{output_file}', allow_pickle=True)")

    logging.info("\n" + "="*70)
    logging.info("✓ Pipeline completed successfully!")
    logging.info("="*70)

    return transformed_embeddings


if __name__ == "__main__":
    transformed = main()
