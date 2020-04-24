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
"""Integration tests for TFX-on-KFP and GCP services."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import kfp
import tensorflow as tf

from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_kubeflow_gcp
from tfx.orchestration import data_types
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.kubeflow import test_utils

# The endpoint of the KFP instance.
# This test fixture assumes an established KFP instance authenticated via
# inverse proxy.
_KFP_ENDPOINT = os.environ['KFP_E2E_ENDPOINT']

# Timeout for a single pipeline run. Set to 10 hours.
_TIME_OUT_SECONDS = 36000

# The base container image name to use when building the image used in tests.
_BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

# The project id to use to run tests.
_GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']

# The GCP region in which the end-to-end test is run.
_GCP_REGION = os.environ['KFP_E2E_GCP_REGION']

# The GCP bucket to use to write output artifacts.
_BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

# The location of test user module file.
# It is retrieved from inside the container subject to testing.
_MODULE_FILE = '/tfx-src/tfx/examples/chicago_taxi_pipeline/taxi_utils.py'

# Update ai_platform_training_args if distributed training was enabled.
# Number of worker machines used in distributed training.
_worker_count = data_types.RuntimeParameter(
    name='worker-count',
    default=2,
    ptype=int,
)

# Type of worker machines used in distributed training.
_worker_type = data_types.RuntimeParameter(
    name='worker-type',
    default='standard',
    ptype=str,
)

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
_ai_platform_serving_args = {
    'model_name': 'chicago_taxi',
    'project_id': _GCP_PROJECT_ID,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model
    'regions': [_GCP_REGION],
}

_beam_pipeline_args = beam_pipeline_args = [
    '--runner=DataflowRunner',
    '--experiments=shuffle_mode=auto',
    '--project=' + _GCP_PROJECT_ID,
    '--temp_location=gs://' + os.path.join(_BUCKET_NAME, 'dataflow', 'tmp'),
    '--region=' + _GCP_REGION,
    '--disk_size_gb=50',
]


class KubeflowGcpPerfTest(test_utils.BaseKubeflowTest):

  @classmethod
  def setUpClass(cls):
    super(test_utils.BaseKubeflowTest, cls).setUpClass()
    # Create a container image for use by test pipelines.
    base_container_image = _BASE_CONTAINER_IMAGE

    cls._container_image = '{}:{}'.format(base_container_image,
                                          cls._random_id())
    cls._build_and_push_docker_image(cls._container_image)

  @classmethod
  def tearDownClass(cls):
    super(test_utils.BaseKubeflowTest, cls).tearDownClass()

  def _compile_and_run_pipeline(self,
                                pipeline: tfx_pipeline.Pipeline,
                                **kwargs):
    """Compiles and runs a KFP pipeline.

    In this method, provided TFX pipeline will be submitted via kfp.Client()
    instead of from Argo.

    Args:
      pipeline: The logical pipeline to run.
      **kwargs: Key-value pairs of runtime paramters passed to the pipeline
        execution.
    """
    client = kfp.Client(host=_KFP_ENDPOINT)

    pipeline_name = pipeline.pipeline_info.pipeline_name
    config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=self._get_kubeflow_metadata_config(),
        tfx_image=self._container_image)
    kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)

    file_path = os.path.join(self._test_dir, '{}.tar.gz'.format(pipeline_name))
    self.assertTrue(tf.io.gfile.exists(file_path))

    run_result = client.create_run_from_pipeline_package(
        pipeline_file=file_path, arguments=kwargs)
    run_id = run_result.run_id
    # DO NOT SUBMIT
    # Seems like wait_for_run_completion is flaky due to lack of retry
    # Any pod unavailability in KFP causes failure, while, the job is still
    # running.
    response = client.wait_for_run_completion(
        run_id=run_id, timeout=_TIME_OUT_SECONDS)
    print(response)
    self.assertEqual(response.run.status.lower(), 'succeeded')

  # def testPrimitiveEnd2EndPipeline(self):
  #   pipeline_name = 'gcp-perf-test-primitive-e2e-test-{}'.format(
  #       self._random_id())
  #   components = test_utils.create_primitive_type_components(pipeline_name)
  #   pipeline = self._create_pipeline(pipeline_name, components)
  #   self._compile_and_run_pipeline(pipeline=pipeline)

  def testFullTaxiGcpPipeline(self):
    pipeline_name = 'gcp-perf-test-full-e2e-test-{}'.format(
        self._random_id())

    # A dict which contains the training job parameters to be passed to Google
    # Cloud AI Platform. For the full set of parameters supported by Google
    # Cloud AI Platform, refer to
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
    ai_platform_training_args = {
        'project': _GCP_PROJECT_ID,
        'region': _GCP_REGION,
        'scaleTier': 'CUSTOM',
        'masterType': 'large_model',
        'masterConfig': {
            'imageUri': self._container_image
        },
        'workerType': _worker_type,
        'parameterServerType': 'standard',
        'workerCount': _worker_count,
        'parameterServerCount': 1
    }

    pipeline = taxi_pipeline_kubeflow_gcp.create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        module_file=_MODULE_FILE,
        ai_platform_training_args=ai_platform_training_args,
        ai_platform_serving_args=_ai_platform_serving_args,
        beam_pipeline_args=_beam_pipeline_args)
    self._compile_and_run_pipeline(
        pipeline=pipeline,
        query_sample_rate=0.1)


if __name__ == '__main__':
  tf.test.main()
