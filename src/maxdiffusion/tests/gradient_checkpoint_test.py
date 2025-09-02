"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest
from types import SimpleNamespace
from absl.testing import absltest

import jax

from maxdiffusion.models.gradient_checkpoint import GradientCheckpointType

class GradientCheckpointTest(unittest.TestCase):
    """Unit test suite for GradientCheckpointType policies."""

    def test_none_policy(self):
        policy = GradientCheckpointType.from_str("NONE")
        self.assertEqual(policy.to_jax_policy(), "skip")

    def test_full_policy(self):
        policy = GradientCheckpointType.from_str("FULL")
        self.assertIsNone(policy.to_jax_policy())

    def test_matmul_without_batch_policy(self):
        policy = GradientCheckpointType.from_str("MATMUL_WITHOUT_BATCH")
        jax_policy_fn = policy.to_jax_policy()
        self.assertIs(jax_policy_fn, jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims)

    def test_offload_matmul_without_batch_policy(self):
        """
        Tests the offload variant by checking the class name of the return value.
        """
        policy = GradientCheckpointType.from_str("OFFLOAD_MATMUL_WITHOUT_BATCH")
        jax_policy_fn = policy.to_jax_policy()
        self.assertTrue(callable(jax_policy_fn))

    def test_custom_policy(self):
        """
        Tests the custom policy by checking the class name of the return value.
        """
        policy = GradientCheckpointType.from_str("CUSTOM")
        names_to_offload = ["attn_output"]
        jax_policy_fn = policy.to_jax_policy(names_which_can_be_offloaded=names_to_offload)
        self.assertTrue(callable(jax_policy_fn))

if __name__ == "__main__":
    absltest.main()