# Copyright 2023 Kevin Herro
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit/
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A scalar value with support for automatic differentiation."""

from typing import Optional, Tuple, Union


class Tensor:
  """
  Represents a scalar value (tensor) with support for automatic differentiation.
  """

  def __init__(
    self,
    data: Union[int, float],
    creators: Optional[list] = None,
    creation_op: Optional['Function'] = None,
  ):
    """
    Initialize a Tensor object.

    :param data: Initial data for the tensor.
    :param creators: List of tensors used to create this tensor (if any).
    :param creation_op: The operation that created this tensor (if any).
    """
    if isinstance(data, float):
      self.data = data
    else:
      raise TypeError('Tensor data must be float')

    self.creators = creators
    self.creation_op = creation_op
    self.grad: Optional[float] = None

  def backward(self, grad: float):
    """
    Performs a backward pass to compute gradients.

    :param grad: Gradient to back-propagate.
    """
    if not isinstance(grad, float):
      raise TypeError('Gradient must be float')

    # Accumulate gradients.
    self.grad = grad if self.grad is None else self.grad + grad

    # If this tensor was created by an operation,
    # propagate the gradient to the creators.
    if self.creation_op is not None:
      grads = self.creation_op.backward(grad)
      for i, creator in enumerate(self.creators):
        creator.backward(grads[i])

  def __add__(self, other: 'Tensor') -> 'Tensor':
    """
    Overloads the addition operator to add two tensors.

    :param other: Another tensor to add.
    :return: A new tensor resulting from the addition.
    """
    return Add()(self, other)

  def __repr__(self) -> str:
    """
    Represents the tensor as a string for debugging purposes.

    :return: String representation of the Tensor object.
    """
    return (
      f'Tensor(data={self.data}, '
      f'creators={self.creators}, '
      f'creation_op={self.creation_op})'
    )

class Function:
  """
  Base class for functions that can be applied to tensors.
  """

  def __call__(self, *tensors: Tensor) -> Tensor:
    """
    Makes the function callable and computes the forward pass.

    :param tensors: Input tensors.
    :return: A tensor resulting from the computation.
    """
    self.tensors = tensors
    self.output = self.forward(*[t.data for t in tensors])
    return Tensor(self.output, creators=list(tensors), creation_op=self)

  def forward(self, *inputs: float) -> float:
    """
    Defines the forward pass of the function.

    :param inputs: Input scalar.
    :return: Output scalar.
    """
    raise NotImplementedError

  def backward(self, grad: float) -> Union[float, Tuple[float, ...]]:
    """
    Defines the backward pass of the function.

    :param grad: Gradient to back-propagate.
    :return: Gradients for inputs.
    """
    raise NotImplementedError


class Add(Function):
  """
  Represents an addition operation for tensors.
  """

  def forward(self, x: float, y: float) -> float:
    """
    Performs the forward pass of addition.

    :param x: First input scalar value.
    :param y: Second input scalar value.
    :return: Sum of ``x`` and ``y``.
    """
    return x + y

  def backward(self, grad: float) -> Tuple[float, ...]:
    """
    Backward pass for addition, simply passing the gradient to both inputs.

    :param grad: Gradient from the next layer.
    :return: Tuple of gradients for each input.
    """
    return grad, grad
