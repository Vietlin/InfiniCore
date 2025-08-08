from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided
import random

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch

def index_copy_inplace(
    output: np.ndarray,
    input: np.ndarray,
    index: np.ndarray,
    dim: int,
):
    torch_output = torch.from_numpy(output) * 0
    torch_output.index_copy_(dim, torch.from_numpy(index), torch.from_numpy(input))
    return torch_output.detach().numpy()


class IndexCopyInplaceTestCase(InfiniopTestCase):
    def __init__(
        self,
        output: np.ndarray,
        shape_output: List[int],
        stride_output: List[int] | None,
        input: np.ndarray,
        shape_input: List[int],
        stride_input: List[int] | None,
        index: np.ndarray,
        shape_index: List[int],
        stride_index: List[int] | None,
        dim: int,
    ):
        super().__init__("index_copy_inplace")
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.index = index
        self.shape_index = shape_index
        self.stride_index = stride_index
        self.dim = dim

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        test_writer.add_array(test_writer.gguf_key("index.shape"), self.shape_index)
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_output)
        

        if self.stride_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input))

        if self.stride_index is not None:
            test_writer.add_array(test_writer.gguf_key("index.strides"), gguf_strides(*self.stride_index))


        test_writer.add_uint64(test_writer.gguf_key("dim"), self.dim)
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            # gguf_strides(*self.stride_output if self.stride_output is not None else contiguous_gguf_strides(self.shape_output))
            gguf_strides(*contiguous_gguf_strides(self.shape_output))
        )        
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("index"), self.index, raw_dtype=np_dtype_to_ggml(self.index.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=np_dtype_to_ggml(self.output.dtype)
        )        

        ans = index_copy_inplace(
            self.output.astype(np.float64),
            self.input.astype(np.float64),
            self.index,
            self.dim,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("index_copy_inplace.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape
        ([13, 1], [13, 1], [4, 1], 1),
        ([13, 4], [13, 4], [10, 1], 1),
        ([13, 4], [15, 4], None, 0),
        ([1333, 4], [1333, 4], [4, 1], 1),
        ([1333, 13], [1333, 13], [1, 1333], 1),
        ([133, 23, 53], [133, 33, 53], None, 1),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for input_shape, output_shape, stride_input, dim in _TEST_CASES_:
            input = np.random.rand(*input_shape).astype(dtype)

            index_list = list(range(output_shape[dim]))
            random.shuffle(index_list)
            index = np.array(index_list[:input_shape[dim]], dtype=np.int64)

            # output = np.empty(output_shape, dtype=dtype)
            output = np.zeros(output_shape, dtype=dtype)

            test_case = IndexCopyInplaceTestCase(
                output=output,
                shape_output=output_shape,
                stride_output=None,
                input=input,
                shape_input=input_shape,
                stride_input=stride_input,
                index=index,
                shape_index=[input_shape[dim],],
                stride_index=(1,),
                dim=dim,
            )
            test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
