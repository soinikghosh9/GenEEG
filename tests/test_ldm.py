"""
Test Latent Diffusion Model (LDM)

Validates LDM U-Net architecture, conditioning, and DDPM scheduler.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.ldm import LatentDiffusionUNetEEG, DDPMScheduler
from configs.model_config import ModelConfig


def test_ldm_instantiation():
    print("[TEST] LDM Instantiation...")
    config = ModelConfig()
    
    ldm = LatentDiffusionUNetEEG(
        latent_channels=config.ldm_latent_channels,
        base_unet_channels=config.ldm_base_unet_channels,
        time_emb_dim=config.ldm_time_emb_dim,
        num_classes=config.ldm_num_classes,
        channel_mults=config.ldm_channel_mults,
        num_res_blocks=config.ldm_num_res_blocks,
        use_cross_attention=config.ldm_use_cross_attention,
        use_adagn=config.ldm_use_adagn,
        context_dim=config.ldm_context_dim,
        feature_cond_dim=config.ldm_feature_cond_dim,
        use_feature_cond=config.ldm_use_feature_cond
    )
    
    total_params = sum(p.numel() for p in ldm.parameters())
    trainable_params = sum(p.numel() for p in ldm.parameters() if p.requires_grad)
    
    print(f"  [SUCCESS] LDM created with {total_params:,} parameters ({trainable_params:,} trainable)")
    return ldm


def test_ldm_forward_pass():
    print("[TEST] LDM Forward Pass...")
    config = ModelConfig()
    
    ldm = LatentDiffusionUNetEEG(
        latent_channels=config.ldm_latent_channels,
        base_unet_channels=config.ldm_base_unet_channels,
        time_emb_dim=config.ldm_time_emb_dim,
        num_classes=config.ldm_num_classes,
        channel_mults=config.ldm_channel_mults,
        num_res_blocks=config.ldm_num_res_blocks,
        use_cross_attention=config.ldm_use_cross_attention,
        use_adagn=config.ldm_use_adagn,
        context_dim=config.ldm_context_dim,
        feature_cond_dim=config.ldm_feature_cond_dim,
        use_feature_cond=config.ldm_use_feature_cond
    )
    
    batch = 4
    latent_channels = config.ldm_latent_channels
    latent_length = 64  # After VAE encoding
    
    z_t = torch.randn(batch, latent_channels, latent_length)
    time_steps = torch.randint(0, 1000, (batch,))
    class_labels = torch.randint(0, config.ldm_num_classes, (batch,))
    feature_cond = torch.randn(batch, config.ldm_feature_cond_dim)
    
    # Forward pass
    output = ldm(z_t, time_steps, class_labels, feature_cond)
    
    assert output.shape == z_t.shape, f"Output shape: {output.shape} vs input {z_t.shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values!"
    
    print(f"  [SUCCESS] Input: {z_t.shape} -> Output: {output.shape}")
    print(f"  Time steps: {time_steps.tolist()}")


def test_ldm_unconditional():
    print("[TEST] LDM Unconditional (no class/feature conditioning)...")
    config = ModelConfig()
    
    ldm = LatentDiffusionUNetEEG(
        latent_channels=config.ldm_latent_channels,
        base_unet_channels=config.ldm_base_unet_channels,
        time_emb_dim=config.ldm_time_emb_dim,
        num_classes=config.ldm_num_classes,
        use_cross_attention=False,  # Disable cross-attention
        use_adagn=False  # Disable AdaGN
    )
    
    batch = 2
    latent_channels = config.ldm_latent_channels
    latent_length = 64
    
    z_t = torch.randn(batch, latent_channels, latent_length)
    time_steps = torch.randint(0, 1000, (batch,))
    
    # Forward pass without conditioning
    output = ldm(z_t, time_steps)
    
    assert output.shape == z_t.shape, f"Output shape mismatch"
    assert torch.isfinite(output).all(), "Output contains non-finite values!"
    
    print(f"  [SUCCESS] Unconditional generation works")


def test_ddpm_scheduler():
    print("[TEST] DDPM Scheduler...")
    
    # Test cosine schedule
    scheduler_cosine = DDPMScheduler(num_timesteps=1000, beta_schedule='cosine')
    
    assert scheduler_cosine.betas.shape == (1000,), f"Betas shape: {scheduler_cosine.betas.shape}"
    assert (scheduler_cosine.betas >= 0.0001).all(), "Betas too small"
    assert (scheduler_cosine.betas <= 0.9999).all(), "Betas too large"
    
    print(f"  [SUCCESS] Cosine schedule: beta range [{scheduler_cosine.betas.min():.6f}, {scheduler_cosine.betas.max():.6f}]")
    
    # Test linear schedule
    scheduler_linear = DDPMScheduler(num_timesteps=1000, beta_schedule='linear', 
                                     beta_start=0.0001, beta_end=0.02)
    
    assert scheduler_linear.betas.shape == (1000,), f"Betas shape: {scheduler_linear.betas.shape}"
    print(f"  [SUCCESS] Linear schedule: beta range [{scheduler_linear.betas.min():.6f}, {scheduler_linear.betas.max():.6f}]")


def test_ddpm_add_noise():
    print("[TEST] DDPM Add Noise (forward diffusion)...")
    
    scheduler = DDPMScheduler(num_timesteps=1000, beta_schedule='cosine')
    
    batch = 4
    channels = 64
    length = 64
    
    x_0 = torch.randn(batch, channels, length)
    t = torch.randint(0, 1000, (batch,))
    
    # Add noise
    x_t = scheduler.add_noise(x_0, t)
    
    assert x_t.shape == x_0.shape, f"Noisy shape mismatch"
    assert torch.isfinite(x_t).all(), "Noisy data contains non-finite values!"
    
    # Check that noise increases with t
    t_early = torch.zeros(batch, dtype=torch.long)
    t_late = torch.full((batch,), 999, dtype=torch.long)
    
    x_t_early = scheduler.add_noise(x_0, t_early)
    x_t_late = scheduler.add_noise(x_0, t_late)
    
    mse_early = torch.mean((x_0 - x_t_early) ** 2).item()
    mse_late = torch.mean((x_0 - x_t_late) ** 2).item()
    
    assert mse_late > mse_early, "Noise should increase with timestep"
    
    print(f"  [SUCCESS] Noise added correctly")
    print(f"  MSE at t=0: {mse_early:.6f}, MSE at t=999: {mse_late:.6f}")


def test_ddpm_step():
    print("[TEST] DDPM Step (reverse diffusion)...")
    
    scheduler = DDPMScheduler(num_timesteps=1000, beta_schedule='cosine')
    
    batch = 4
    channels = 64
    length = 64
    
    x_0 = torch.randn(batch, channels, length)
    t = torch.full((batch,), 500, dtype=torch.long)
    
    # Add noise
    noise = torch.randn_like(x_0)
    x_t = scheduler.add_noise(x_0, t, noise)
    
    # Denoise step (assuming perfect prediction)
    x_t_minus_1 = scheduler.step(noise, t, x_t)
    
    assert x_t_minus_1.shape == x_0.shape, f"Denoised shape mismatch"
    assert torch.isfinite(x_t_minus_1).all(), "Denoised data contains non-finite values!"
    
    print(f"  [SUCCESS] Denoising step works")


def test_ldm_gradient_flow():
    print("[TEST] LDM Gradient Flow...")
    config = ModelConfig()
    
    ldm = LatentDiffusionUNetEEG(
        latent_channels=config.ldm_latent_channels,
        base_unet_channels=config.ldm_base_unet_channels,
        time_emb_dim=config.ldm_time_emb_dim,
        num_classes=config.ldm_num_classes,
        channel_mults=config.ldm_channel_mults,
        num_res_blocks=config.ldm_num_res_blocks,
        use_cross_attention=config.ldm_use_cross_attention,
        use_adagn=config.ldm_use_adagn,
        context_dim=config.ldm_context_dim
    )
    
    batch = 2
    latent_channels = config.ldm_latent_channels
    latent_length = 64
    
    z_t = torch.randn(batch, latent_channels, latent_length, requires_grad=True)
    time_steps = torch.randint(0, 1000, (batch,))
    class_labels = torch.randint(0, config.ldm_num_classes, (batch,))
    
    # Forward pass
    pred_noise = ldm(z_t, time_steps, class_labels)
    
    # Simple loss
    target_noise = torch.randn_like(pred_noise)
    loss = torch.mean((pred_noise - target_noise) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and torch.isfinite(p.grad).all() for p in ldm.parameters())
    assert has_gradients, "No gradients or non-finite gradients!"
    
    print(f"  [SUCCESS] Gradient flow verified. Loss: {loss.item():.6f}")


def test_ldm_with_scheduler():
    print("[TEST] LDM with Scheduler Integration...")
    config = ModelConfig()
    
    ldm = LatentDiffusionUNetEEG(
        latent_channels=config.ldm_latent_channels,
        base_unet_channels=config.ldm_base_unet_channels,
        time_emb_dim=config.ldm_time_emb_dim,
        num_classes=config.ldm_num_classes
    )
    
    scheduler = DDPMScheduler(num_timesteps=1000, beta_schedule='cosine')
    
    batch = 2
    latent_channels = config.ldm_latent_channels
    latent_length = 64
    
    # Simulate training step
    x_0 = torch.randn(batch, latent_channels, latent_length)
    t = torch.randint(0, 1000, (batch,))
    noise = torch.randn_like(x_0)
    
    # Forward diffusion
    x_t = scheduler.add_noise(x_0, t, noise)
    
    # Predict noise
    class_labels = torch.randint(0, config.ldm_num_classes, (batch,))
    pred_noise = ldm(x_t, t, class_labels)
    
    # Loss
    loss = torch.mean((pred_noise - noise) ** 2)
    
    assert torch.isfinite(loss), "Loss is not finite!"
    
    print(f"  [SUCCESS] LDM + Scheduler integration works. Loss: {loss.item():.6f}")


def main():
    print("="*60)
    print("Testing Latent Diffusion Model")
    print("="*60)
    
    test_ldm_instantiation()
    test_ldm_forward_pass()
    test_ldm_unconditional()
    test_ddpm_scheduler()
    test_ddpm_add_noise()
    test_ddpm_step()
    test_ldm_gradient_flow()
    test_ldm_with_scheduler()
    
    print("="*60)
    print("[SUCCESS] All LDM tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()
