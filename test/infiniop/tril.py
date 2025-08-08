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
    ((5, 6), 0),
    ((4, 5), -1),
    ((6, 7), 2),
    ((5, 2), 3),
    ((8, 8), -2),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
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

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_tril(output, input, diagonal):
    torch.tril(input, diagonal, out=output)

    

def test(
    handle,
    device,
    x_shape,
    diagonal,
    inplace,
    dtype,
    sync=None,
):

    output = TestTensor(
        x_shape,
        None,
        # x_stride,
        dtype,
        device,
        # "manual",
        # set_tensor=None
    )

    input = TestTensor(
        x_shape,
        None,
        # x_stride,
        dtype,
        device,
        # "manual",
        # set_tensor=None
    )


    print(
        # TODO:
        f"Testing tril on {InfiniDeviceNames[device]} with diagonal:{diagonal}"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    torch_tril(output.torch_tensor(), input.torch_tensor(), diagonal)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTrilDescriptor(
            handle,
            ctypes.byref(descriptor),
			output.descriptor,
			input.descriptor,
			diagonal,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [output,input]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTrilWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    #TODO:
    workspace = TestWorkspace(workspace_size.value, input.device)

    def lib_tril():
        check_error(
            LIBINFINIOP.infiniopTril(
                descriptor,
                workspace.data(),
                workspace.size(),
				output.data(),
				input.data(),                
                None,
            )
        )

    lib_tril()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        # TODO:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)     
    # TODO:       
    # print("x:", input.torch_tensor())
    print("CALCULATED:\n", output.actual_tensor(), )
    print("GT\n", output.torch_tensor())
    # TODO:
    assert torch.allclose(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        # TODO:
        profile_operation("PyTorch", lambda: torch_tril(
            output.torch_tensor(), input.torch_tensor(), diagonal
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_tril(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyTrilDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my tril passed!\033[0m")
