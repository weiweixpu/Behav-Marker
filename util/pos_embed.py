import torch
import numpy as np

def get_2d_sincos_pos_embed(
        embed_dim: int,
        grid_size: int,
        cls_token: bool = False
) -> np.ndarray:
    """Generates 2D sine-cosine positional embeddings.

    Args:
        embed_dim: Dimension of the output embeddings
        grid_size: Spatial grid size (height/width of the grid)
        cls_token: Whether to prepend a classification token placeholder

    Returns:
        Positional embeddings with shape:
        - [grid_size*grid_size, embed_dim] when cls_token=False
        - [1 + grid_size*grid_size, embed_dim] when cls_token=True
    """
    # Generate coordinate grid
    grid_rows = np.arange(grid_size, dtype=np.float32)
    grid_cols = np.arange(grid_size, dtype=np.float32)
    col_grid, row_grid = np.meshgrid(grid_cols, grid_rows)  # x, y coordinates

    # Stack coordinates into [2, grid_size, grid_size] array
    spatial_coords = np.stack([col_grid, row_grid], axis=0)

    # Add singleton dimension for broadcasting compatibility
    spatial_coords = spatial_coords.reshape((2, 1, grid_size, grid_size))

    # Generate embeddings from coordinate grid
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, spatial_coords)

    # Prepend classification token placeholder if needed
    if cls_token:
        cls_token_embed = np.zeros((1, embed_dim), dtype=np.float32)
        pos_embed = np.concatenate([cls_token_embed, pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed