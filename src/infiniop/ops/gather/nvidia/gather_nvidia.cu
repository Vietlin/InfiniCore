#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "gather_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"

namespace op::gather::nvidia {

//  ---------------------- start: launchKernel: call kernel function of CUDA -----------------------
template <unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL launchKernel(
    char * output,
    const char * input,
    const int64_t * index,
    size_t ndim,
    size_t element_size,
    size_t index_gather_size,
    ptrdiff_t * output_strides,
    ptrdiff_t * input_strides,
    ptrdiff_t * index_strides,
    ptrdiff_t * contiguous_strides,
    int dim,
    cudaStream_t stream
) {
    gatherKernel<BLOCK_SIZE>(
        output,
        input,
        index,
        ndim,
        element_size,
        index_gather_size,
        output_strides,
        input_strides,
        index_strides,
        contiguous_strides,
        dim,
        stream
    );
}
//  ----------------------- end: launchKernel: call kernel function of CUDA ------------------------

//  ----------------------------------- start: call launchKernel -----------------------------------
template<unsigned int BLOCK_SIZE>
infiniStatus_t calculate_gather(
    const GatherInfo &info,
    void * output,
    const void * input,
    const int64_t *  index,
    cudaStream_t stream,
    void * workspace
) {
    size_t ndim = info.ndim;
    ptrdiff_t * contiguous_strides = new ptrdiff_t[ndim];
    size_t last_dim = 1, last_stride = 1;
    size_t gather_dim = info.dim;
    for(size_t d = 0; d < ndim; d ++)
    {
        if (d == gather_dim) 
            continue;        
        contiguous_strides[d] = last_dim * last_stride;
        last_dim = info.output_shape[d];
        last_stride = contiguous_strides[d];
    }
    size_t batch_size = last_dim * last_stride;


    ptrdiff_t * contiguous_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace);
    ptrdiff_t * input_strides_cuda = contiguous_strides_cuda + ndim;
    ptrdiff_t * output_strides_cuda = input_strides_cuda + ndim;
    ptrdiff_t * index_strides_cuda = output_strides_cuda + ndim;

    CHECK_CUDA(cudaMemcpyAsync(contiguous_strides_cuda, contiguous_strides, sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(input_strides_cuda, info.input_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_strides_cuda, info.output_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(index_strides_cuda, info.index_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));    

    
    launchKernel<BLOCK_SIZE><<<batch_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<char *>(output),
        reinterpret_cast<const char *>(input),
        index,
        ndim,
        infiniSizeOf(info.dtype),
        info.output_shape[gather_dim],
        output_strides_cuda,
        input_strides_cuda,
        index_strides_cuda,
        contiguous_strides_cuda,
        info.dim,
        stream
    );
    return INFINI_STATUS_SUCCESS;
}
//  ------------------------------------ end: call launchKernel ------------------------------------


struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t index_desc,
    size_t dim
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = output_desc->dtype();
    size_t WorkSpaceSize = sizeof(ptrdiff_t) * input_desc->ndim() * 4;
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = GatherInfo::createGatherInfo(
        output_desc,
        input_desc,
        index_desc,
        dim
    );
    CHECK_RESULT(result);
    const GatherInfo &info = result.take();
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        new Opaque{handle->internal()},
        handle->device, handle->device_id
    );    
    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void * workspace,
    size_t workspace_size,
    void * output,
    const void * input,
    const void * index,
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    #define CALCULATE_GATHER_WITH_BLOCK_SIZE(BLOCK_SIZE) \
        return calculate_gather<BLOCK_SIZE>(_info, output, input, (const int64_t *)index, stream, workspace);

    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024)
        CALCULATE_GATHER_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512)
        CALCULATE_GATHER_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    else
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    #undef CALCULATE_GATHER_WITH_BLOCK_SIZE

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::gather::nvidia
