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

_TEST_CASES_ = [
#    ((3, 4), 3, 0),
#    ((2,  2,), 2, 1),
#    ((2, 5, 3, 4), 2, 0),
#    ((2, 3, 4), 3, 1),
#    ((2, 6, 4), 6, 1),

#    ((1, 8, 1), 20, 1),
   ((1, 20, 1), 20, 1),
# TODO:
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]


# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]
# _TENSOR_DTYPES = [InfiniDtype.F32]

# Tolerance map for different data types

# _TOLERANCE_MAP = {
#     InfiniDtype.F16: {"atol": 0, "rtol": 0},
#     InfiniDtype.F32: {"atol": 0, "rtol": 0},
#     InfiniDtype.BF16: {"atol": 0, "rtol": 0},
# }
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_index_copy_inplace(output, input, index, dim):
    output.index_copy_(dim, index, input)
    

def test(
    handle,
    device,
    input_shape,
    output_dim_size,
    dim,

    # x_stride,
    # dim,
    # index_shape,
    inplace,
    dtype,
    sync=None,
):
    print(
        # TODO:
        f"Testing index_copy_inplace on {InfiniDeviceNames[device]} with shape:{input_shape},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    # torch_dtype = {InfiniDtype.F16: torch.half, InfiniDtype.F32: torch.float, InfiniDtype.BF16: torch.bfloat16}

    # torch_output = None
    output_shape = list(input_shape)
    output_shape[dim] = output_dim_size
    output = TestTensor(
        output_shape,
        None,
        # torch_output.stride(),
        dtype,
        device,
        "zeros",
        # set_tensor=torch_output
    )

    torch_input = torch.randint(low=0, high = 16, size=input_shape, dtype=output._torch_tensor.dtype)
    
    input = TestTensor(
        input_shape,
        # None,
        torch_input.stride(),
        dtype,
        device,
        "manual",
        set_tensor=torch_input
    )

    # torch_index = torch.randint(low=0, high=output_dim_size, size=[input_shape[dim]], dtype=torch.int64)
    index_list = list(range(output_shape[dim]))
    import random
    random.shuffle(index_list)
    torch_index = torch.tensor(index_list[:input_shape[dim]], dtype=torch.int64)
    index = TestTensor(
        [input_shape[dim]],
        # None,
        torch_index.stride(),
        InfiniDtype.I64,
        device,
        "manual",
        set_tensor=torch_index
    )




    torch_index_copy_inplace(output.torch_tensor(), input.torch_tensor(), index.torch_tensor(), dim)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateIndexCopyInplaceDescriptor(
            handle,
            ctypes.byref(descriptor),
			output.descriptor,
			input.descriptor,
			index.descriptor,
			dim,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [output, input, index]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetIndexCopyInplaceWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    #TODO:
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_index_copy_inplace():
        check_error(
            LIBINFINIOP.infiniopIndexCopyInplace(
                descriptor,
                workspace.data(),
                workspace.size(),
				output.data(),
				input.data(),
				index.data(),                
                None,
            )
        )

    lib_index_copy_inplace()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
# TODO:
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
    # TODO:
    print('input:\n', input.torch_tensor())
    print('index:\n', index.torch_tensor())
    print('output:\n', output.torch_tensor(), '\n', output.actual_tensor(), )


    assert torch.allclose(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        # TODO:
        profile_operation("PyTorch", lambda: torch_index_copy_inplace(
            output, input, index, dim
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_index_copy_inplace(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyIndexCopyInplaceDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my index_copy_inplace passed!\033[0m")
