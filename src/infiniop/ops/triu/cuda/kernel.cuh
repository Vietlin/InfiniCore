#ifndef __TRIU_KERNEL_CUH__
#define __TRIU_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
__device__ void triuKernel(
    char * output,
    const char * input,
    int row_size,
    int column_size,
    size_t elem_size,
    int diagonal,
    cudaStream_t stream
) {
    int row = blockIdx.x;
    auto output_ptr = output + row * column_size * elem_size;
    auto input_ptr = input + row * column_size * elem_size;
    if (column_size - 1 < row + diagonal)
        cudaMemsetAsync(
            output_ptr,
            0,
            elem_size * column_size,
            stream
        );
    else if (0 >= row + diagonal)
        cudaMemcpyAsync(
            output_ptr,
            input_ptr, 
            elem_size * column_size,
            cudaMemcpyDeviceToDevice,
            stream
        );
    else {
        cudaMemsetAsync(
            output_ptr,
            0,
            elem_size * (row + diagonal),
            stream
        );
        cudaMemcpyAsync(
            output_ptr + elem_size * (row + diagonal),
            input_ptr + elem_size * (row + diagonal),
            elem_size * (column_size - row - diagonal),
            cudaMemcpyDeviceToDevice,
            stream
        ); 
    }  
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __TRIU_KERNEL_CUH__
