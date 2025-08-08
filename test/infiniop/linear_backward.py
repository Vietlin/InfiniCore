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
#    ((2, 3, 4), None, 1, (2, 3, 2),),
    (5, 8, True), 
    (7, 3, True), 
    (20, 10, True), 
    (20, 10, False), 
# TODO:
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

    
def torch_linear_backward(grad_x, grad_w, grad_b, grad_y, x, w, b, bias:bool):
    #TODO:
    # return torch.nn.functional.linear(x, w, b)
    ...
    x.requires_grad_(True)
    w.requires_grad_(True)
    if bias:
        b.requires_grad_(True)
    y = torch.nn.functional.linear(x, w, bias=(b if bias else None))
    y.backward(grad_y)
    grad_x.copy_(x.grad)
    grad_w.copy_(w.grad)
    if bias:
        grad_b.copy_(b.grad)



    

def test(
    handle,
    device,
    in_features,
    out_features,
    bias,
    dtype,
    sync=None,
):
    print(
        # TODO:
        f"Testing linear_backward on {InfiniDeviceNames[device]} with "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    # torch_dtype = {InfiniDtype.F16: torch.half, InfiniDtype.F32: torch.float, InfiniDtype.BF16: torch.bfloat16}

    # torch_grad_x = None
    grad_x = TestTensor(
        [in_features],
        None,
        dtype,
        device,
    )

    # torch_grad_w = None
    grad_w = TestTensor(
        [out_features, in_features],
        None,
        dtype,
        device,
    )

    # torch_grad_b = None
    grad_b = TestTensor(
        [out_features] if bias else [],
        None,
        dtype,
        device,
    ) if bias else None

    # torch_grad_y = None
    grad_y = TestTensor(
        [out_features],
        None,
        # torch_grad_y.stride(),
        dtype,
        device,
        # "ones",
        # "manual",
        # set_tensor=torch_grad_y
    )

    # torch_x = None
    x = TestTensor(
        [in_features],
        None,
        # torch_x.stride(),
        dtype,
        device,
        # "ones",
        # "manual",
        # set_tensor=torch_x
    )

    # torch_w = None
    w = TestTensor(
        [out_features, in_features],
        None,
        # torch_w.stride(),
        dtype,
        device,
        # "ones",
        # "manual",
        # set_tensor=torch_w
    )

    b = TestTensor(
        [out_features] if bias else [],
        None,
        # torch_w.stride(),
        dtype,
        device,
        # "ones",
        # "manual",
        # set_tensor=torch_w
    )        




    torch_linear_backward(grad_x.torch_tensor(), grad_w.torch_tensor(),
        grad_b.torch_tensor() if bias else None,
        grad_y.torch_tensor(), x.torch_tensor(), w.torch_tensor(), b.torch_tensor(), bias)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLinearBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
			grad_x.descriptor,
			grad_w.descriptor,
			(grad_b.descriptor if bias else None),
			grad_y.descriptor,
			x.descriptor,
			w.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [grad_x, grad_w, grad_b, grad_y, x, w, b]:
        if tensor is not None:
            tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLinearBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    #TODO:
    workspace = TestWorkspace(workspace_size.value, grad_x.device)

    def lib_linear_backward():
        check_error(
            LIBINFINIOP.infiniopLinearBackward(
                descriptor,
                workspace.data(),
                workspace.size(),
				grad_x.data(),
				grad_w.data(),
				grad_b.data() if bias else None,
				grad_y.data(),
				x.data(),
				w.data(),                
                None,
            )
        )

    lib_linear_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
# TODO:
    if DEBUG:
        debug(grad_x.actual_tensor(), grad_x.torch_tensor(), atol=atol, rtol=rtol)
        debug(grad_w.actual_tensor(), grad_w.torch_tensor(), atol=atol, rtol=rtol)
        debug(grad_b.actual_tensor(), grad_b.torch_tensor(), atol=atol, rtol=rtol)
    # TODO:
    print('grad_y:\n', grad_y.torch_tensor())
    print('x:\n', x.torch_tensor())
    print('w:\n', w.torch_tensor())
    if bias:
        print('b:\n', b.torch_tensor())
    print('grad_x:\n', grad_x.torch_tensor(), '\n', grad_x.actual_tensor(), )
    print('grad_w:\n', grad_w.torch_tensor(), '\n', grad_w.actual_tensor(), )
    if bias:
        print('grad_b:\n', grad_b.torch_tensor(), '\n', grad_b.actual_tensor(), )


    assert torch.allclose(grad_x.actual_tensor(), grad_x.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(grad_w.actual_tensor(), grad_w.torch_tensor(), atol=atol, rtol=rtol)
    if bias:
        assert torch.allclose(grad_b.actual_tensor(), grad_b.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        # TODO:
        profile_operation("PyTorch", lambda: torch_linear_backward(
            grad_x, grad_w, grad_b if bias else None, grad_y, x, w,bias
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_linear_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyLinearBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my linear_backward passed!\033[0m")
