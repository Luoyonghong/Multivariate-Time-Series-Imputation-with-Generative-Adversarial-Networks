#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 23:13:29 2018

@author: yonghong, luo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import RNNCell
import tensorflow as tf
if "1.5" in tf.__version__ or "1.7" in tf.__version__:
    from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
    from tensorflow.python.layers import base as base_layer
    from tensorflow.python.ops import nn_ops
    _BIAS_VARIABLE_NAME = "bias"
    _WEIGHTS_VARIABLE_NAME = "kernel"
    
    class MyGRUCell15(LayerRNNCell):
    #todo: 直接改就行了，已经试验过了，不影响dynamic_rnn的调用
      """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    
      Args:
        num_units: int, The number of units in the GRU cell.
        activation: Nonlinearity to use.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.  If not `True`, and the existing scope already has
         the given variables, an error is raised.
        kernel_initializer: (optional) The initializer to use for the weight and
        projection matrices.
        bias_initializer: (optional) The initializer to use for the bias.
        name: String, the name of the layer. Layers with the same name will
          share weights, but to avoid mistakes we require reuse=True in such
          cases.
      """
      def __init__(self,
                   num_units,
                   activation=None,
                   reuse=None,
                   kernel_initializer=None,
                   bias_initializer=None,
                   name=None):
        super(MyGRUCell15, self).__init__(_reuse=reuse, name=name)
    
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
    
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
    
      @property
      def state_size(self):
        return self._num_units
    
      @property
      def output_size(self):
        return self._num_units
    
      def build(self, inputs_shape):
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)
    
        input_depth = inputs_shape[1].value-self._num_units
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))
    
        self.built = True
    
      def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        totalLength=inputs.get_shape().as_list()[1]
        inputs_=inputs[:,0:totalLength-self._num_units]
        rth=inputs[:,totalLength-self._num_units:]
        inputs=inputs_
        state=math_ops.multiply(rth,state)
        
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)
    
        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    
        r_state = r * state
    
        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)
    
        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h


elif "1.4" in tf.__version__:
    from tensorflow.python.ops.rnn_cell_impl import _Linear
elif "1.2" in tf.__version__:
    from tensorflow.python.ops.rnn_cell_impl import _linear



class MyGRUCell4(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(MyGRUCell4, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    # inputs = realinputs + m +rt
    # rt's length is self._num_units
    # state = rt * older state 
    # input = first 2 part
    totalLength=inputs.get_shape().as_list()[1]
    inputs_=inputs[:,0:totalLength-self._num_units]
    rth=inputs[:,totalLength-self._num_units:]
    inputs=inputs_
    state=math_ops.multiply(rth,state)
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    value = math_ops.sigmoid(self._gate_linear([inputs, state]))
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    if self._candidate_linear is None:
      with vs.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    c = self._activation(self._candidate_linear([inputs, r_state]))
    new_h = u * state + (1 - u) * c
    return new_h, new_h


class MyGRUCell2(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(MyGRUCell2, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    # inputs = realinputs + m +rt
    # rt's length is self._num_units
    # state = rt * older state 
    # input = first 2 part
    totalLength=inputs.get_shape().as_list()[1]
    inputs_=inputs[:,0:totalLength-self._num_units]
    rth=inputs[:,totalLength-self._num_units:]
    inputs=inputs_
    state=math_ops.multiply(rth,state)
    with vs.variable_scope("gates"):  # Reset gate and update gate.
      # We start with bias of 1.0 to not reset and not update.
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        dtype = [a.dtype for a in [inputs, state]][0]
        bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
      value = math_ops.sigmoid(
          _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                  self._kernel_initializer))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    with vs.variable_scope("candidate"):
      c = self._activation(
          _linear([inputs, r * state], self._num_units, True,
                  self._bias_initializer, self._kernel_initializer))
    new_h = u * state + (1 - u) * c
    return new_h, new_h
