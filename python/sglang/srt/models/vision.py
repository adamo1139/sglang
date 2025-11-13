"""
DP sharding helpers for multimodal vision models.

Adapted from vLLM vision utilities.
"""

import itertools
import math
from typing import List, Literal, Tuple

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)


def get_load_balance_assignment(
    sizes: List[int],
    num_gpus: int = 2,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Greedy load balancing by total size per GPU.

    Args:
        sizes: Size per sample (e.g., patches per image).
        num_gpus: Number of devices to balance across.

    Returns:
        shuffle_indices: indices to reorder inputs by assigned device.
        gpu_sample_counts: number of samples on each device.
        grouped_sizes_per_gpu: total size on each device.
    """

    n_samples = len(sizes)
    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    gpu_assignments: List[List[int]] = [[] for _ in range(num_gpus)]
    gpu_loads: List[int] = [0] * num_gpus

    large_to_small_indices = sorted(range(n_samples), key=lambda i: sizes[i], reverse=True)
    for idx in large_to_small_indices:
        gid = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[gid].append(idx)
        gpu_loads[gid] += sizes[idx]

    shuffle_indices: List[int] = []
    gpu_sample_counts: List[int] = []
    for gid in range(num_gpus):
        shuffle_indices.extend(gpu_assignments[gid])
        gpu_sample_counts.append(len(gpu_assignments[gid]))

    return shuffle_indices, gpu_sample_counts, gpu_loads


def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw_list: List[List[int]],
    *,
    rope_type: Literal["rope_3d", "rope_2d"],
):
    """
    Shard variable-length vision inputs across TP ranks for data-parallel execution,
    gather outputs, and restore original order.

    Args:
        vision_model: Vision encoder module.
        pixel_values: Flattened patch tensor [total_patches, C].
        grid_thw_list: Per-sample grid (T, H, W) or (H, W) depending on model.
        rope_type: "rope_3d" (e.g. Qwen2.5-VL) or "rope_2d".

    Returns:
        Tuple[torch.Tensor, ...] of per-sample embeddings (in original order).
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank_local = get_tensor_model_parallel_rank()

    patches_per_image = [int(math.prod(g)) for g in grid_thw_list]
    cum_patches = [0, *itertools.accumulate(patches_per_image)]

    image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len = get_load_balance_assignment(
        patches_per_image, tp_size
    )
    cum_gpu_counts = [0, *itertools.accumulate(gpu_sample_counts)]
    local_indices = image_to_tp_rank[cum_gpu_counts[tp_rank_local] : cum_gpu_counts[tp_rank_local] + gpu_sample_counts[tp_rank_local]]

    if len(local_indices) > 0:
        pixel_values_local = torch.cat(
            [pixel_values[cum_patches[i] : cum_patches[i + 1]] for i in local_indices]
        )
    else:
        pixel_values_local = torch.empty(
            (0, pixel_values.shape[1]), device=pixel_values.device, dtype=pixel_values.dtype
        )

    if rope_type == "rope_2d":
        # 2D rope models reduce embed dim by kernel product
        embed_dim_reduction_factor = (
            vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
        )
    else:
        embed_dim_reduction_factor = (
            vision_model.spatial_merge_size * vision_model.spatial_merge_size
        )

    max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor if tp_size > 0 else 0
    local_grid_thw_list = [grid_thw_list[i] for i in local_indices]

    if rope_type == "rope_2d":
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, torch.tensor(local_grid_thw_list))
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
    else:
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, local_grid_thw_list)
        else:
            image_embeds_local = torch.empty(
                (0, getattr(vision_model, "out_hidden_size", pixel_values_local.shape[-1])),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )

    current_len = image_embeds_local.shape[0]
    if current_len < max_len_per_rank:
        padding_size = max_len_per_rank - current_len
        if rope_type == "rope_2d":
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1], image_embeds_local.shape[2]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        else:
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
    else:
        image_embeds_local_padded = image_embeds_local

    gathered = tensor_model_parallel_all_gather(image_embeds_local_padded, dim=0)

    rank_embeddings: List[torch.Tensor] = []
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (grouped_pixel_values_len[rank] // embed_dim_reduction_factor)
        rank_embeddings.append(gathered[start_idx:end_idx])

    patches_per_output_image = [p // embed_dim_reduction_factor for p in patches_per_image]

    original_order_embeddings: List[torch.Tensor] = [None] * len(grid_thw_list)  # type: ignore
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            rank_images = image_to_tp_rank[current_idx : current_idx + count]
            rank_embed = rank_embeddings[rank]
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[embed_start : embed_start + img_patches]
                embed_start += img_patches
            current_idx += count

    out_embeddings = tuple(embed for embed in original_order_embeddings if embed is not None)
    assert len(out_embeddings) == len(original_order_embeddings), "Found unassigned embeddings"
    return out_embeddings

