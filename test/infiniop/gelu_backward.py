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


def torch_gelu_backward(grad_input, input:torch.Tensor, grad_output):
    #TODO:
    input.requires_grad_(True)
    output = torch.nn.functional.gelu(input, approximate="tanh")
    output.backward(grad_output)
    return input.grad
    

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
        f"Testing gelu_backward on {InfiniDeviceNames[device]} with "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    # torch_dtype = {InfiniDtype.F16: torch.half, InfiniDtype.F32: torch.float, InfiniDtype.BF16: torch.bfloat16}

    # torch_grad_input = None
    grad_input = TestTensor(
        input_shape,
        None,
        # torch_grad_input.stride(),
        dtype,
        device,
        # "manual",
        # set_tensor=torch_grad_input
    )

    torch_input = torch.rand(input_shape) * 10 - 5

    input = TestTensor(
        input_shape,
        # None,
        torch_input.stride(),
        dtype,
        device,
        "manual",
        set_tensor=torch_input
    )

    # torch_grad_output = None
    grad_output = TestTensor(
        input_shape,
        None,
        # torch_grad_output.stride(),
        dtype,
        device,
        # "manual",
        # set_tensor=torch_grad_output
    )



    grad_input._torch_tensor = torch_gelu_backward(grad_input.torch_tensor(), input.torch_tensor(), grad_output.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGeLUBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_input.descriptor,
            input.descriptor,
            grad_output.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [grad_input, input, grad_output]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGeLUBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    #TODO:
    workspace = TestWorkspace(workspace_size.value, grad_input.device)

    def lib_gelu_backward():
        check_error(
            LIBINFINIOP.infiniopGeLUBackward(
                descriptor,
                workspace.data(),
                workspace.size(),
                grad_input.data(),
                input.data(),
                grad_output.data(),                
                None,
            )
        )

    lib_gelu_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
# TODO:
    if DEBUG:
        debug(grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol)
    # TODO:
    print('input:\n', input.torch_tensor())
    print('grad_output:\n', grad_output.torch_tensor())
    print('grad_input:\n', grad_input.torch_tensor(), '\n', grad_input.actual_tensor(), )


    assert torch.allclose(grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        # TODO:
        profile_operation("PyTorch", lambda: torch_gelu_backward(
            grad_input, input, grad_output
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gelu_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyGeLUBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my gelu_backward passed!\033[0m")
