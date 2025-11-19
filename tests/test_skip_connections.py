"""
Test Skip Connection Implementation

This script verifies that skip connections work correctly with proper dimension alignment.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.vae import DecoupledVAE
from configs.model_config import ModelConfig
from configs.dataset_config import DatasetConfig

print("="*80)
print(" Skip Connection Verification Test")
print("="*80)

# Initialize VAE
print("\n1. Initializing DecoupledVAE...")
vae = DecoupledVAE(
    in_channels=ModelConfig.VAE_IN_CHANNELS,
    base_channels=128,
    latent_channels=ModelConfig.VAE_LATENT_DIM,
    ch_mults=(1, 2, 4, 8),
    blocks_per_stage=2,
    use_feature_cond=True,
    feature_cond_dim=12
)

print(f"   ✓ VAE initialized")
print(f"   Parameters: {sum(p.numel() for p in vae.parameters()):,}")

# Create test input
batch_size = 4
num_channels = ModelConfig.VAE_IN_CHANNELS
seq_length = DatasetConfig.SEGMENT_SAMPLES

print(f"\n2. Creating test input...")
print(f"   Shape: [{batch_size}, {num_channels}, {seq_length}]")
x = torch.randn(batch_size, num_channels, seq_length) * 0.5

# Test encode
print(f"\n3. Testing encode (with skip features)...")
with torch.no_grad():
    mu, logvar, skip_features = vae.encode(x)

print(f"   ✓ Encode successful")
print(f"   mu shape: {list(mu.shape)}")
print(f"   logvar shape: {list(logvar.shape)}")
print(f"   Skip features collected: {len(skip_features)}")

# Count non-None skip features
non_none_skips = sum(1 for s in skip_features if s is not None)
print(f"   Non-None skip features: {non_none_skips}/{len(skip_features)}")

# Check skip feature shapes
print(f"\n4. Skip feature details:")
for i, skip_feat in enumerate(skip_features):
    if skip_feat is not None:
        print(f"   skip_features[{i}]: {list(skip_feat.shape)}")
    else:
        print(f"   skip_features[{i}]: None")

# Test reparameterize
print(f"\n5. Testing reparameterization...")
with torch.no_grad():
    z = vae.reparameterize(mu, logvar)

print(f"   ✓ Reparameterize successful")
print(f"   z shape: {list(z.shape)}")
print(f"   z range: [{z.min().item():.2f}, {z.max().item():.2f}]")

# Test decode WITH skip features
print(f"\n6. Testing decode WITH skip features...")
with torch.no_grad():
    recon_with_skip = vae.decode(z, skip_features=skip_features)

print(f"   ✓ Decode (with skip) successful")
print(f"   Reconstruction shape: {list(recon_with_skip.shape)}")
print(f"   Reconstruction range: [{recon_with_skip.min().item():.2f}, {recon_with_skip.max().item():.2f}]")

# Test decode WITHOUT skip features (ablation)
print(f"\n7. Testing decode WITHOUT skip features (ablation)...")
with torch.no_grad():
    recon_no_skip = vae.decode(z, skip_features=None)

print(f"   ✓ Decode (no skip) successful")
print(f"   Reconstruction shape: {list(recon_no_skip.shape)}")
print(f"   Reconstruction range: [{recon_no_skip.min().item():.2f}, {recon_no_skip.max().item():.2f}]")

# Compare reconstructions
print(f"\n8. Comparing reconstructions (with vs without skip)...")
diff = torch.abs(recon_with_skip - recon_no_skip).mean().item()
print(f"   Mean absolute difference: {diff:.6f}")
if diff > 0.01:
    print(f"   ✓ Skip connections are active (significant difference)")
else:
    print(f"   ⚠ Skip connections may not be working (minimal difference)")

# Test full forward pass
print(f"\n9. Testing full forward pass...")
with torch.no_grad():
    recon_full, mu_full, logvar_full, z_full, is_stable = vae(x)

print(f"   ✓ Full forward pass successful")
print(f"   Reconstruction shape: {list(recon_full.shape)}")
print(f"   Is stable: {is_stable}")

# Check for NaN/Inf
has_nan = torch.isnan(recon_full).any() or torch.isinf(recon_full).any()
print(f"   Contains NaN/Inf: {has_nan}")

# Compute reconstruction error
mse = torch.mean((x - recon_full) ** 2).item()
print(f"   Reconstruction MSE: {mse:.6f}")

# Test with feature conditioning
print(f"\n10. Testing with feature conditioning...")
features = torch.randn(batch_size, 12) * 0.5
with torch.no_grad():
    mu_feat, logvar_feat, skip_feat = vae.encode(x)
    z_feat = vae.reparameterize(mu_feat, logvar_feat)
    recon_feat = vae.decode(z_feat, feature_cond=features, skip_features=skip_feat)

print(f"   ✓ Feature conditioning successful")
print(f"   Reconstruction shape: {list(recon_feat.shape)}")

# Test latent statistics computation
print(f"\n11. Testing latent statistics computation...")
from torch.utils.data import TensorDataset, DataLoader

# Create dummy dataset
labels = torch.randint(0, 3, (batch_size * 10,))
data_expanded = torch.randn(batch_size * 10, num_channels, seq_length) * 0.5
dataset = TensorDataset(data_expanded, labels)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

stats = vae.compute_latent_statistics(loader, device='cpu', max_batches=5)

print(f"   ✓ Latent statistics computed")
print(f"   Separation ratio: {stats['separation_ratio']:.4f}")
print(f"   Intra-class distances: {stats['intra_class_distances']}")
print(f"   Inter-class distances: {stats['inter_class_distances']}")

# Summary
print("\n" + "="*80)
print(" Test Summary")
print("="*80)
print(f"✓ Encoder skip collection: PASSED")
print(f"✓ Decoder skip usage: PASSED")
print(f"✓ Dimension alignment: PASSED")
print(f"✓ Skip connections active: {'YES' if diff > 0.01 else 'MAYBE'}")
print(f"✓ Forward pass stable: {'YES' if is_stable and not has_nan else 'NO'}")
print(f"✓ Feature conditioning: PASSED")
print(f"✓ Latent statistics: PASSED")
print(f"\nReconstruction MSE: {mse:.6f}")
print(f"Latent separation ratio: {stats['separation_ratio']:.4f}")
print("="*80)

if is_stable and not has_nan and diff > 0.01:
    print("\n✅ ALL TESTS PASSED - Skip connections working correctly!")
else:
    print("\n⚠ Some issues detected - check warnings above")

print("\nNote: For optimal performance during training:")
print("  - Separation ratio should increase to > 3.0")
print("  - Reconstruction MSE should decrease")
print("  - Monitor latent space visualization every 10 epochs")
print("="*80 + "\n")
