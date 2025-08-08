#ifndef __INDEX_COPY_INPLACE_KERNEL_CUH__
#define __INDEX_COPY_INPLACE_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
__device__ void indexCopyInplaceKernel(
    char * output,
    const char * input,
    const int64_t * index,
    size_t element_size,
    size_t index_len,
    ptrdiff_t index_stride,
    ptrdiff_t meta_stride,
    cudaStream_t stream
) {
    size_t dst_index = blockIdx.x;
    size_t src_index = index_len - 1;
    size_t copy_unit_size = element_size * meta_stride;
    while (true)
    {
        if(*(index + src_index * index_stride) == dst_index)
        {
            cudaMemcpyAsync(
                output + element_size * dst_index * meta_stride,
                input + element_size * src_index * meta_stride,
                copy_unit_size,
                cudaMemcpyDeviceToDevice,
                stream
            );
            break;
        }
        else if (src_index == 0)
            break;
        src_index --;
    }
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __INDEX_COPY_INPLACE_KERNEL_CUH__
