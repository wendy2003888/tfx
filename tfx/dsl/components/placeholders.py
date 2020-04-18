# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains structures that describe components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text


class InputValuePlaceholder(object):
  """Will be replaced with the string value of the input argument."""

  def __init__(self, input_name: Text):
    self.input_name = input_name


class InputUriPlaceholder(object):
  """Will be replaced with the URI of the input argument."""

  def __init__(self, input_name: Text):
    self.input_name = input_name


class OutputUriPlaceholder(object):
  """Will be replaced with the URI for the output."""

  def __init__(self, output_name: Text):
    self.output_name = output_name
