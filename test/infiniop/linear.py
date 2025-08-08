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
    # (1, 1, True),
    # (5, 4, True),
    # (5, 1, True),
    (5, 4, False),
    (3, 9, False),
    # (20, 10, True),
]



# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]
# _TENSOR_DTYPES = [InfiniDtype.F32]

# Tolerance map for different data types

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_linear(x, w, b, bias):
    # return torch.nn.functional.linear(x.type(torch.float), w.type(torch.float), b.type(torch.float))
    return torch.nn.functional.linear(x, w, bias=(b if bias else None))
    

def test(
    handle,
    device,
    in_features,
    out_features,
    bias,
    dtype,
    sync=None,
):
    # torch_dtype = {InfiniDtype.F16: torch.half, InfiniDtype.F32: torch.float, InfiniDtype.BF16: torch.bfloat16}
    y = TestTensor(
        [out_features],
        None,
        # x_stride,
        dtype,
        device,
        # "manual",
        # set_tensor=None
    )
    # torch_x = torch.arange(in_features).type(torch_dtype[dtype])
    x = TestTensor(
        [in_features],
        None,
        # torch_x.stride(),
        dtype,
        device,
        # "manual",
        # set_tensor=torch_x
    )
    # torch_w = torch.arange(in_features * out_features).reshape([out_features, in_features]).type(torch_dtype[dtype])
    w = TestTensor(
        [out_features, in_features],
        None,
        # torch_w.stride(),
        dtype,
        device,
        # "manual",
        # set_tensor=torch_w,
    )
    # torch_b = (torch.zeros([out_features]) + 0.5).type(torch_dtype[dtype])
    b = TestTensor(
        [out_features],
        None,
        # torch_b.stride(),
        dtype,
        device,
        # "manual",
        # set_tensor=torch_b
    ) if bias else None

    print(
        # TODO:
        f"Testing linear on {InfiniDeviceNames[device]} with shape: {w._torch_tensor.shape}, bias:{bias},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    y._torch_tensor = torch_linear(x.torch_tensor(), w.torch_tensor(), (b.torch_tensor() if bias else None), bias)#.type(torch_dtype[dtype])

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLinearDescriptor(
            handle,
            ctypes.byref(descriptor),
			y.descriptor,
			x.descriptor,
			w.descriptor,
			(b.descriptor if bias else None),
        )
    )
    
    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [y, x, w] + ([b] if bias else []):
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLinearWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    #TODO:
    workspace = TestWorkspace(workspace_size.value, x.device)
    
    def lib_linear():
        check_error(
            LIBINFINIOP.infiniopLinear(
                descriptor,
                workspace.data(),
                workspace.size(),
				y.data(),
				x.data(),
				w.data(),
				b.data() if bias else None,                
                None,
            )
        )
    
    lib_linear()
    

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)     
    # TODO:       
    print("x:\n", x.torch_tensor())
    print("w:\n", w.torch_tensor())
    if bias:
        print("b:\n", b.torch_tensor())

    print("CALCULATED:\n", y.actual_tensor(), )
    print("GT\n", y.torch_tensor())
    # TODO:
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        # TODO:
        profile_operation("PyTorch", lambda: torch_linear(
            y, x, w, (b if bias else None), bias
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_linear(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my linear passed!\033[0m")
