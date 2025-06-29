import triton
import triton.language as tl

@triton.jit
def streamk_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak, #
    stride_bk, stride_bn, #
    stride_cm, stride_cn, #
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, #
    t0: tl.constexpr, # tile index offset (phase start)
    units: tl.constexpr, # number of units of work for stream-k
):
    # Program and grid identification
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)

    # Tile grid dimensions and sub-dot counts
    grid_n = tl.cdiv(N, BLOCK_SIZE_N) # tiles along N
    n_sub = tl.cdiv(K, BLOCK_SIZE_K) # number of sub-dot blocks per tile

    # Loop over the program's assigned work units
    for u in range(pid, units, num_pids):
        tile = t0 + u // n_sub # index of the tile
        k_loc = u % n_sub # sub-dot index

        # Decode tile coordinates
        pid_m = tile // grid_n
        pid_n = tile % grid_n

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
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
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
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, P, mask=c_mask)