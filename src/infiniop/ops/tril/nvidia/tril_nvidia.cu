#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "tril_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"

namespace op::tril::nvidia {

INFINIOP_CUDA_KERNEL launchKernel(
    char * output,
    const char * input,
    int row_size,
    int column_size,
    size_t elem_size,
    int diagonal,
    cudaStream_t stream
) {
    trilKernel(
        output,
        input,
        row_size,
        column_size,
        elem_size,
        diagonal,
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
    int diagonal
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = output_desc->dtype();
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = TrilInfo::createTrilInfo(
        output_desc,
        input_desc,
        diagonal
    );
    CHECK_RESULT(result);
    const TrilInfo &info = result.take();
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
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    launchKernel<<<_info.row_size, 1, 0, stream>>>(
        reinterpret_cast<char *>(output),
        reinterpret_cast<const char *>(input),
        _info.row_size,
        _info.column_size,
        _info.elem_size,        
        _info.diagonal,
        stream
    );    
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::tril::nvidia
