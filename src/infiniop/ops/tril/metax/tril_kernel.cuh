#ifndef __TRIL_KERNEL_CUH__
#define __TRIL_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
__device__ void trilKernel(
    char * output,
    const char * input,
    int row_size,
    int column_size,
    size_t elem_size,
    int diagonal,
    hcStream_t stream
) {
    int row = blockIdx.x;
    auto output_ptr = output + row * column_size * elem_size;
    auto input_ptr = input + row * column_size * elem_size;
    if (0 > row + diagonal)
        hcMemsetAsync(
            output_ptr,
            0,
            elem_size * column_size,
            stream
        );
    else if (column_size - 1 <= row + diagonal)
        hcMemcpyAsync(
            output_ptr,
            input_ptr, 
            elem_size * column_size,
            hcMemcpyDeviceToDevice,
            stream
        );
    else {
        hcMemsetAsync(
            output_ptr + elem_size * (row + diagonal + 1),
            0,
            elem_size * (column_size - row - diagonal - 1),
            stream
        );
        hcMemcpyAsync(
            output_ptr,
            input_ptr,
            elem_size * (row + diagonal + 1),
            hcMemcpyDeviceToDevice,
            stream
        ); 
    }  
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __TRIL_KERNEL_CUH__