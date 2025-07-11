import torch

import triton
import triton.language as tl

DEVICE = "cuda"

@triton.jit
def add_kernel(x_ptr, # *pointer* to the first input vector.
               y_ptr, # *pointer* to the second input vector.
               output_ptr, # *ponter* to output vector.
               n_elements, # size of the vector.
               BLOCK_SIZE: tl.constexpr, # number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value`
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0
    # This program will process inputs that are offset from the initial data.
    # For instance if you had a vector of length 256 and block_size of 64, the
    # programs would each access elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking any extra elements in case the input is
    # not a multiple of block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x+y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analagous to CUDA launch grids. It can either be Tuple[int], or 
    # Callable(metaparameters)->Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE: 
    #   - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #   - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #   - Don't forget to pass meta-paramters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, 
    # the kernel is still running asynchronously at this point.
    return output 

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)], # Different possible values for `x_name`
        x_log=True, # x-axis is logarithmic
        line_arg='provider', # Argument name whose value corresponds to a different line in the plot
        line_vals=['torch', 'triton'], # Possible values for `line_arg`
        line_names=['Torch', 'Triton'], # label names for the lines
        styles=[('blue', '-'), ('green', '-')], # line styles
        ylabel='GB/s', # label name for the y-axis
        plot_name='vector-add-performance', # Name for the plot. Used also as a file name for saving the plot.
        args={}, # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles=[0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, save_path='/home/connorb/triton_tutorials/vector-addition-results/')