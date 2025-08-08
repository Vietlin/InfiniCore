#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "index_copy_inplace_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../rearrange/nvidia/rearrange_nvidia.cuh"
#include "../info.h"

namespace op::index_copy_inplace::nvidia {

INFINIOP_CUDA_KERNEL launchKernel(
    char * output,
    const char * input,
    const int64_t * index,
    size_t element_size,
    size_t index_len,
    ptrdiff_t index_stride,
    ptrdiff_t meta_stride,
    cudaStream_t stream
) {
    indexCopyInplaceKernel(
        output,
        input,
        index,
        element_size,
        index_len,
        index_stride,
        meta_stride,
        stream
    );

}

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
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = IndexCopyInplaceInfo::createIndexCopyInplaceInfo(
        output_desc,
        input_desc,
        index_desc,
        dim
    );
    CHECK_RESULT(result);
    const IndexCopyInplaceInfo &info = result.take();
    size_t WorkSpaceSize = (info.total_input_size + info.total_output_size) * infiniSizeOf(dtype);

    InfiniopTensorDescriptor * rearrange_in_desc = new InfiniopTensorDescriptor(
        dtype, input_desc->ndim(), input_desc->shape().data(), info.meta_strides.data()
    );
    InfiniopTensorDescriptor * rearrange_out_desc = new InfiniopTensorDescriptor(
        dtype, input_desc->ndim(), output_desc->shape().data(), info.meta_strides.data()
    );        
    
    void * in_rearrange_descriptor = nullptr;
    void * out_rearrange_descriptor = nullptr;

    op::rearrange::nvidia::Descriptor::create(
        handle_, reinterpret_cast<op::rearrange::nvidia::Descriptor **>(&in_rearrange_descriptor),
        rearrange_in_desc, input_desc
    );
    op::rearrange::nvidia::Descriptor::create(
        handle_, reinterpret_cast<op::rearrange::nvidia::Descriptor **>(&out_rearrange_descriptor),
        output_desc, rearrange_out_desc
    );    

    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        new Opaque{handle->internal()},
        handle->device, handle->device_id,
        in_rearrange_descriptor,
        out_rearrange_descriptor        
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

    size_t size_of_dtype = infiniSizeOf(_info.dtype);
    char* workspace_in = reinterpret_cast<char*>(workspace);
    char* workspace_out = workspace_in + size_of_dtype * _info.total_input_size;
    CHECK_STATUS(reinterpret_cast<op::rearrange::nvidia::Descriptor *>(_rearrange_desc_in)->calculate(workspace_in, input, stream));
    cudaMemsetAsync(workspace_out, 0, _info.total_output_size * size_of_dtype, stream);
    cudaDeviceSynchronize();
    launchKernel<<<_info.output_shape[_info.dim], 1, 0, stream>>>(
        reinterpret_cast<char*>(workspace_out),
        reinterpret_cast<char*>(workspace_in),
        reinterpret_cast<const int64_t*>(index),
        size_of_dtype,
        _info.index_shape[0],
        _info.index_strides[0],
        _info.meta_strides[_info.dim],
        stream
    );
    cudaDeviceSynchronize();

    CHECK_STATUS(reinterpret_cast<op::rearrange::nvidia::Descriptor *>(_rearrange_desc_out)->calculate(output, workspace_out, stream));
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::index_copy_inplace::nvidia
