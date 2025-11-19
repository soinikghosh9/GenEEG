"""
Test Building Blocks

Quick validation that all building block components are working correctly.
Tests import, instantiation, and forward pass with dummy data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.building_blocks import (
    TemporalSelfAttention1D,
    CrossAttention,
    AdaGN,
    ResNetBlock,
    SpatialChannelAttention,
    ChannelAttention,
    EnhancedTemporalAttention,
    Downsample1D,
    Upsample1D,
    sinusoidal_time_embedding
)


def test_temporal_self_attention():
    print("[TEST] TemporalSelfAttention1D...")
    channels = 64
    length = 256
    batch = 2
    
    attn = TemporalSelfAttention1D(channels=channels, num_heads=8)
    x = torch.randn(batch, channels, length)
    out = attn(x)
    
    assert out.shape == (batch, channels, length), f"Shape mismatch: {out.shape}"
    print(f"  [SUCCESS] Input: {x.shape} -> Output: {out.shape}")


def test_cross_attention():
    print("[TEST] CrossAttention...")
    query_dim = 128
    context_dim = 64
    seq_len = 256
    context_len = 32
    batch = 2
    
    cross_attn = CrossAttention(query_dim=query_dim, context_dim=context_dim, n_heads=8, head_dim=16)
    x = torch.randn(batch, seq_len, query_dim)
    context = torch.randn(batch, context_len, context_dim)
    out = cross_attn(x, context)
    
    assert out.shape == (batch, seq_len, query_dim), f"Shape mismatch: {out.shape}"
    print(f"  [SUCCESS] Query: {x.shape}, Context: {context.shape} -> Output: {out.shape}")


def test_adagn():
    print("[TEST] AdaGN...")
    channels = 64
    length = 256
    cond_dim = 128
    batch = 2
    
    adagn = AdaGN(num_groups=8, num_channels=channels, cond_dim=cond_dim)
    x = torch.randn(batch, channels, length)
    cond = torch.randn(batch, cond_dim)
    out = adagn(x, cond)
    
    assert out.shape == (batch, channels, length), f"Shape mismatch: {out.shape}"
    print(f"  [SUCCESS] Input: {x.shape}, Cond: {cond.shape} -> Output: {out.shape}")


def test_resnet_block():
    print("[TEST] ResNetBlock (vanilla)...")
    in_ch, out_ch = 64, 128
    length = 256
    batch = 2
    
    block = ResNetBlock(in_channels=in_ch, out_channels=out_ch)
    x = torch.randn(batch, in_ch, length)
    out = block(x)
    
    assert out.shape == (batch, out_ch, length), f"Shape mismatch: {out.shape}"
    print(f"  [SUCCESS] Input: {x.shape} -> Output: {out.shape}")


def test_resnet_block_with_time():
    print("[TEST] ResNetBlock (with time embedding)...")
    in_ch, out_ch = 64, 128
    length = 256
    time_emb_dim = 256
    batch = 2
    
    block = ResNetBlock(in_channels=in_ch, out_channels=out_ch, time_emb_dim=time_emb_dim)
    x = torch.randn(batch, in_ch, length)
    t_emb = torch.randn(batch, time_emb_dim)
    out = block(x, t_emb=t_emb)
    
    assert out.shape == (batch, out_ch, length), f"Shape mismatch: {out.shape}"
    print(f"  [SUCCESS] Input: {x.shape}, Time: {t_emb.shape} -> Output: {out.shape}")


def test_resnet_block_with_adagn():
    print("[TEST] ResNetBlock (with AdaGN)...")
    in_ch, out_ch = 64, 128
    length = 256
    class_cond_dim = 128
    batch = 2
    
    block = ResNetBlock(
        in_channels=in_ch, 
        out_channels=out_ch, 
        use_adagn=True, 
        class_cond_dim=class_cond_dim
    )
    x = torch.randn(batch, in_ch, length)
    c_emb = torch.randn(batch, class_cond_dim)
    out = block(x, c_emb=c_emb)
    
    assert out.shape == (batch, out_ch, length), f"Shape mismatch: {out.shape}"
    print(f"  [SUCCESS] Input: {x.shape}, Class: {c_emb.shape} -> Output: {out.shape}")


def test_downsample_upsample():
    print("[TEST] Downsample1D and Upsample1D...")
    channels = 64
    length = 256
    batch = 2
    
    down = Downsample1D(channels)
    up = Upsample1D(channels)
    
    x = torch.randn(batch, channels, length)
    x_down = down(x)
    x_up = up(x_down)
    
    assert x_down.shape == (batch, channels, length // 2), f"Downsample shape: {x_down.shape}"
    assert x_up.shape == (batch, channels, length), f"Upsample shape: {x_up.shape}"
    print(f"  [SUCCESS] Input: {x.shape} -> Down: {x_down.shape} -> Up: {x_up.shape}")


def test_spatial_channel_attention():
    print("[TEST] SpatialChannelAttention...")
    channels = 64
    length = 256
    batch = 2
    
    attn = SpatialChannelAttention(channels)
    x = torch.randn(batch, channels, length)
    out = attn(x)
    
    assert out.shape == (batch, channels, length), f"Shape mismatch: {out.shape}"
    print(f"  [SUCCESS] Input: {x.shape} -> Output: {out.shape}")


def test_channel_attention():
    print("[TEST] ChannelAttention...")
    channels = 64
    length = 256
    batch = 2
    
    attn = ChannelAttention(channels, reduction=8)
    x = torch.randn(batch, channels, length)
    out = attn(x)
    
    assert out.shape == (batch, channels, length), f"Shape mismatch: {out.shape}"
    print(f"  [SUCCESS] Input: {x.shape} -> Output: {out.shape}")


def test_enhanced_temporal_attention():
    print("[TEST] EnhancedTemporalAttention...")
    channels = 64
    length = 256
    batch = 2
    
    attn = EnhancedTemporalAttention(channels, num_heads=8)
    x = torch.randn(batch, channels, length)
    out = attn(x)
    
    assert out.shape == (batch, channels, length), f"Shape mismatch: {out.shape}"
    print(f"  [SUCCESS] Input: {x.shape} -> Output: {out.shape}")


def test_sinusoidal_time_embedding():
    print("[TEST] sinusoidal_time_embedding...")
    batch = 8
    embed_dim = 256
    
    t = torch.randint(0, 1000, (batch,))
    emb = sinusoidal_time_embedding(t, embed_dim)
    
    assert emb.shape == (batch, embed_dim), f"Shape mismatch: {emb.shape}"
    print(f"  [SUCCESS] Timesteps: {t.shape} -> Embeddings: {emb.shape}")


def main():
    print("="*60)
    print("Testing Building Blocks")
    print("="*60)
    
    test_temporal_self_attention()
    test_cross_attention()
    test_adagn()
    test_resnet_block()
    test_resnet_block_with_time()
    test_resnet_block_with_adagn()
    test_downsample_upsample()
    test_spatial_channel_attention()
    test_channel_attention()
    test_enhanced_temporal_attention()
    test_sinusoidal_time_embedding()
    
    print("="*60)
    print("[SUCCESS] All building block tests passed!")
    print("="*60)


if __name__ == "__main__":
    main()
