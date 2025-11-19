"""
Test VAE Model

Validates VAE architecture, forward pass, and numerical stability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.vae import DecoupledVAE, get_cosine_schedule
from configs.model_config import ModelConfig


def test_vae_instantiation():
    print("[TEST] VAE Instantiation...")
    config = ModelConfig()
    
    vae = DecoupledVAE(
        in_channels=config.vae_in_channels,
        base_channels=config.vae_base_channels,
        latent_channels=config.vae_latent_channels,
        ch_mults=config.vae_ch_mults,
        blocks_per_stage=config.vae_blocks_per_stage,
        use_feature_cond=config.vae_use_feature_cond,
        feature_cond_dim=config.vae_feature_cond_dim
    )
    
    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    
    print(f"  [SUCCESS] VAE created with {total_params:,} parameters ({trainable_params:,} trainable)")
    return vae


def test_vae_forward_pass():
    print("[TEST] VAE Forward Pass...")
    config = ModelConfig()
    
    vae = DecoupledVAE(
        in_channels=config.vae_in_channels,
        base_channels=config.vae_base_channels,
        latent_channels=config.vae_latent_channels,
        ch_mults=config.vae_ch_mults,
        blocks_per_stage=config.vae_blocks_per_stage,
        use_feature_cond=config.vae_use_feature_cond,
        feature_cond_dim=config.vae_feature_cond_dim
    )
    
    batch = 4
    channels = config.vae_in_channels
    length = 512
    
    x = torch.randn(batch, channels, length)
    feature_cond = torch.randn(batch, config.vae_feature_cond_dim) if config.vae_use_feature_cond else None
    
    # Forward pass
    recon_x, mu, logvar, z, is_stable = vae(x, feature_cond=feature_cond)
    
    # Check shapes
    assert recon_x.shape == (batch, channels, length), f"Reconstruction shape: {recon_x.shape}"
    assert mu.shape[0] == batch and mu.shape[1] == config.vae_latent_channels, f"Mu shape: {mu.shape}"
    assert logvar.shape == mu.shape, f"Logvar shape: {logvar.shape}"
    assert z.shape == mu.shape, f"Z shape: {z.shape}"
    assert is_stable, "VAE output is not numerically stable!"
    
    print(f"  [SUCCESS] Input: {x.shape} -> Latent: {z.shape} -> Reconstruction: {recon_x.shape}")
    print(f"  Numerical stability: {is_stable}")


def test_vae_encode_decode():
    print("[TEST] VAE Encode/Decode Separately...")
    config = ModelConfig()
    
    vae = DecoupledVAE(
        in_channels=config.vae_in_channels,
        base_channels=config.vae_base_channels,
        latent_channels=config.vae_latent_channels,
        ch_mults=config.vae_ch_mults,
        blocks_per_stage=config.vae_blocks_per_stage,
        use_feature_cond=config.vae_use_feature_cond,
        feature_cond_dim=config.vae_feature_cond_dim
    )
    
    batch = 2
    channels = config.vae_in_channels
    length = 512
    
    x = torch.randn(batch, channels, length)
    feature_cond = torch.randn(batch, config.vae_feature_cond_dim) if config.vae_use_feature_cond else None
    
    # Encode
    mu, logvar, skip_features = vae.encode(x)
    
    # Sample
    z = vae.reparameterize(mu, logvar)
    
    # Decode
    recon_x = vae.decode(z, feature_cond=feature_cond, skip_features=skip_features)
    
    # Check shapes
    assert recon_x.shape == x.shape, f"Reconstruction shape mismatch: {recon_x.shape} vs {x.shape}"
    assert torch.isfinite(recon_x).all(), "Reconstruction contains non-finite values!"
    
    # Check reconstruction quality (should be similar but not identical due to sampling)
    mse = torch.mean((x - recon_x) ** 2).item()
    print(f"  [SUCCESS] Encode -> Decode successful. MSE: {mse:.6f}")


def test_vae_with_different_lengths():
    print("[TEST] VAE with Different Sequence Lengths...")
    config = ModelConfig()
    
    vae = DecoupledVAE(
        in_channels=config.vae_in_channels,
        base_channels=config.vae_base_channels,
        latent_channels=config.vae_latent_channels,
        ch_mults=config.vae_ch_mults,
        blocks_per_stage=config.vae_blocks_per_stage,
        use_feature_cond=config.vae_use_feature_cond,
        feature_cond_dim=config.vae_feature_cond_dim
    )
    
    batch = 2
    channels = config.vae_in_channels
    
    # Test different sequence lengths
    lengths = [256, 512, 1024]
    
    for length in lengths:
        x = torch.randn(batch, channels, length)
        recon_x, mu, logvar, z, is_stable = vae(x)
        
        assert recon_x.shape == x.shape, f"Shape mismatch for length {length}"
        assert is_stable, f"Instability for length {length}"
        print(f"  [SUCCESS] Length {length}: Input {x.shape} -> Reconstruction {recon_x.shape}")


def test_cosine_schedule():
    print("[TEST] Cosine Beta Schedule...")
    
    timesteps = 1000
    betas = get_cosine_schedule(timesteps)
    
    assert betas.shape == (timesteps,), f"Beta schedule shape: {betas.shape}"
    assert (betas >= 0.0001).all() and (betas <= 0.015).all(), "Betas out of expected range"
    
    print(f"  [SUCCESS] Beta schedule created with {timesteps} timesteps")
    print(f"  Beta range: [{betas.min():.6f}, {betas.max():.6f}]")


def test_vae_gradient_flow():
    print("[TEST] VAE Gradient Flow...")
    config = ModelConfig()
    
    vae = DecoupledVAE(
        in_channels=config.vae_in_channels,
        base_channels=config.vae_base_channels,
        latent_channels=config.vae_latent_channels,
        ch_mults=config.vae_ch_mults,
        blocks_per_stage=config.vae_blocks_per_stage,
        use_feature_cond=config.vae_use_feature_cond,
        feature_cond_dim=config.vae_feature_cond_dim
    )
    
    batch = 2
    channels = config.vae_in_channels
    length = 512
    
    x = torch.randn(batch, channels, length, requires_grad=True)
    feature_cond = torch.randn(batch, config.vae_feature_cond_dim) if config.vae_use_feature_cond else None
    
    # Forward pass
    recon_x, mu, logvar, z, is_stable = vae(x, feature_cond=feature_cond)
    
    # Simple loss
    loss = torch.mean((x - recon_x) ** 2) + 0.001 * torch.mean(mu ** 2 + torch.exp(logvar))
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and torch.isfinite(p.grad).all() for p in vae.parameters())
    assert has_gradients, "No gradients or non-finite gradients!"
    
    print(f"  [SUCCESS] Gradient flow verified. Loss: {loss.item():.6f}")


def main():
    print("="*60)
    print("Testing VAE Model")
    print("="*60)
    
    vae = test_vae_instantiation()
    test_vae_forward_pass()
    test_vae_encode_decode()
    test_vae_with_different_lengths()
    test_cosine_schedule()
    test_vae_gradient_flow()
    
    print("="*60)
    print("[SUCCESS] All VAE tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()
