import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

_TEST_CASES = [
    ((2, 3, 4), None, 2, (2, 3, 2),),
    ((2, 3, 4), None, 0, (1, 3, 4),),
    ((4, 3, 4), None, 2, (4, 3, 4),),
    # ((2, 3, 4), None, 0.00001),
    # ((4, 3, 2), None, 0.00001),
    # ((15, 16, 17), None, 0.00001),
    # ((5,), None, (1,), 0),
    # ((1, 3), None, (1, 3), 0),
    # ((1, 4), None, (1, 1), 1),
    # ((3, 5), None, (3, 1), 1),
    # ((32, 20, 512), None, (32, 20, 1), 2),
    # ((32, 20, 512), None, (32, 1, 512), 1),
]

# No implement for INPLACE


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]
# _TENSOR_DTYPES = [InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_gather(output, input, dim, index):
    torch.gather(input, dim, index, out=output)

def test(
    handle,
    device,
    x_shape,
    x_stride,
    dim,
    index_shape,
    dtype,
    sync=None,
):
    
    input = TestTensor(
        x_shape,
        x_stride,
        dtype,
        device
    ) 
    torch_index = torch.randint(low=0, high=x_shape[dim], size=index_shape, dtype=torch.int64)
    index = TestTensor(
        index_shape,
        torch_index.stride(),
        InfiniDtype.I64,
        device,
        "manual",
        set_tensor=torch_index
    ) 
    print("index:", index.torch_tensor())
    output = TestTensor(
        index_shape,
        None,
        dtype,
        device,
        "zeros"
    )     




    print(
        f"Testing Gather on {InfiniDeviceNames[device]} with input shape:{x_shape}, dim:{dim}, index_shape: {index_shape}"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    torch_gather(output.torch_tensor(), input.torch_tensor(), dim, index.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGatherDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            input.descriptor,
            index.descriptor,
            dim
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input, output, index]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGatherWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, input.device)

    def lib_gather():
        check_error(
            LIBINFINIOP.infiniopGather(
                descriptor,
                workspace.data(),
                workspace.size(),
                output.data(),
                input.data(),
                index.data(),
                None,
            )
        )

    lib_gather()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:

        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
    print("x:", input.torch_tensor())
    

    print("CALCULATED:\n", output.actual_tensor(), )
    print("GT\n", output.torch_tensor())
    assert torch.allclose(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_gather(
            output.torch_tensor(), input.torch_tensor(), dim, index.torch_tensor()
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gather(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyGatherDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my Gather passed!\033[0m")
