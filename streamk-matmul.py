import triton
import triton.language as tl

import torch

@triton.jit
def streamk_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak, 
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # The work unit ranges and distribution along k-dimension.
        start_unit, # First unit for Stream-K to work on.
        end_unit, # Last unit for Stream-K to work on.
        n_sub, # Number of sub-dot blocks per tile.
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr
):
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    n_pids = tl.num_programs(axis=0)

    for unit_idx in range(start_unit, end_unit, n_pids):
        # Tile and sub-dot block indexing
        tile = unit // n_sub # index of the tile
        tile_m = tile // grid_n
        tile_n = tile % grid_n
        k_loc = unit % n_sub # sub-dot index

        # Determine effective k-block size for final sub-dot
        k0 = k_loc * BLOCK_SIZE_K
        is_last = k_loc == n_sub - 1
        rem_k = K - k0
        cur_bK = tl.where(is_last, rem_k, bK)

        # Create k-axis mask for loads
        k_idx = tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_idx < cur_bK

        # Pointer offsets
        offs_k = (k0 + tl.arange(0, BLOCK_SIZE_K)) % K
        offs_am = (tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn (tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # Load A sub-block with mask on k-dimension
        A_blk = tl.load(
            a_ptrs, 
            mask=k_mask[None, :], 
            other=0.0 
        )
        # Load B sub-block with mask on k-dimension
        B_blk = tl.load(
            b_ptrs,
            mask=k_mask[:, None],
            other=0.0
        )
        # Compute partial product
        P = tl.dot(A_blk, B_blk)    # [bM, bN]

        # Accumulate and write back the output block.
        offs_cm = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, P, mask=c_mask)

@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
):
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def p_matmul(a, b):
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    n_sms = props.multi_processor_count

    print(f"GPU has {num_sms} SMs")

    M, K = a.shape
    K, N = b.shape

    # Compute number of tiles for Stream-K to handle
    n_tiles = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']))
    n_sub = lambda META: triton.cdiv(K, META['BLOCK_SIZE_K'])
    n_rem_tiles = n_tiles % n_sms
    do_add_full_wave = triton.cdiv(n_tiles, n_sms) > 1 and (not n_rem_tiles == 0)
    n_rem_tiles = triton.where(do_add_full_wave, n_rem_tiles + n_sms, n_rem_tiles)
    n_dataparallel_tiles = n_tiles - n_rem_tiles

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 1D launch kernel for data-parallel phase
    matmul_kernel[n_dataparallel_tiles](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    
    streamk_start_unit = n_dataparallel_tiles * n_sub - 1
    n_streamk_units = n_rem_tiles * n_sub
    streamk_end_unit = streamk_start_unit + n_streamk_units

    # 1D launch kernel for Stream-K phase
    streamk_kernel[n_sms](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        streamk_start_unit, #
        streamk_end_unit, #
        n_sub, #
    )

    return c