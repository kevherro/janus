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
"""Tests for `tensor.py`."""

import unittest
from janus._src.tensor import Tensor


class TensorOperationTest(unittest.TestCase):
  """Tests for Tensor operations."""

  def test_addition_forward(self):
    t1 = Tensor(1.0)
    t2 = Tensor(2.0)
    t3 = t1 + t2
    self.assertEqual(t3.data, 3.0)

  def test_addition_backward(self):
    t1 = Tensor(1.0)
    t2 = Tensor(2.0)
    t3 = t1 + t2
    t3.backward(1.0)
    self.assertEqual(t1.data, 1.0)
    self.assertEqual(t2.data, 2.0)

  def test_multiplication_forward(self):
    t1 = Tensor(2.0)
    t2 = Tensor(4.0)
    t3 = t1 * t2
    self.assertEqual(t3.data, 8.0)

  def test_multiplication_backward(self):
    t1 = Tensor(2.0)
    t2 = Tensor(4.0)
    t3 = t1 * t2
    t3.backward(1.0)
    self.assertEqual(t1.grad, 4.0)
    self.assertEqual(t2.grad, 2.0)

if __name__ == '__main__':
  unittest.main()
