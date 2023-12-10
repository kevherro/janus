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
"""Tests for janus."""

import janus
import unittest


class JanusTest(unittest.TestCase):
  """Test janus can be imported correctly."""

  def test_import(self):
    self.assertTrue(hasattr(janus, 'Tensor'))


if __name__ == '__main__':
  unittest.main()
