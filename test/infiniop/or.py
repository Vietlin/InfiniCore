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
    ((5,),),
    ((2,5,),),
    ((2,5,10),),
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


def torch_or(c, a, b):
    #TODO:
    torch.or(input=input, out=output)
    

def test(
    handle,
    device,
    input_shape,
    inplace,
    dtype,
    sync=None,
):
    print(
        # TODO:
        f"Testing or on {InfiniDeviceNames[device]} with "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    # torch_dtype = {InfiniDtype.F16: torch.half, InfiniDtype.F32: torch.float, InfiniDtype.BF16: torch.bfloat16}

    # torch_c = None
    c = TestTensor(
        input_shape,
        None,
        # torch_c.stride(),
        dtype,
        device,
        # "manual",
        # set_tensor=torch_c
    )

    # torch_a = None
    a = TestTensor(
        input_shape,
        None,
        # torch_a.stride(),
        dtype,
        device,
        # "manual",
        # set_tensor=torch_a
    )

    # torch_b = None
    b = TestTensor(
        input_shape,
        None,
        # torch_b.stride(),
        dtype,
        device,
        # "manual",
        # set_tensor=torch_b
    )



    # output._torch_tensor = 
    torch_or(c.torch_tensor(), a.torch_tensor(), b.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateOrDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [c, a, b]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetOrWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    #TODO:
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_or():
        check_error(
            LIBINFINIOP.infiniopOr(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                a.data(),
                b.data(),                
                None,
            )
        )

    lib_or()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
# TODO:
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    # TODO:
    print('a:\n', a.torch_tensor())
    print('b:\n', b.torch_tensor())
    print('c:\n', c.torch_tensor(), '\n', c.actual_tensor(), )


    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        # TODO:
        profile_operation("PyTorch", lambda: torch_or(
            c, a, b
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_or(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyOrDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my or passed!\033[0m")
