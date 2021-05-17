import os
import sys

from lairningdecisions.utils.db import select_record, P_MARKER, _BACKOFFICE_DB

_SHELL = os.getenv('SHELL')
_CONDA_PREFIX = os.getenv('CONDA_PREFIX_1') if 'CONDA_PREFIX_1' in os.environ.keys() else os.getenv('CONDA_PREFIX')

_TRAINER_YAML = lambda cluster_name, cloud_provider: "configs/{}_{}_scaler.yaml".format(cluster_name, cloud_provider)
_TRAINER_PATH = lambda cluster_name, cloud_provider: "trainer_{}_{}".format(cluster_name, cloud_provider)
_CMD_PREFIX = ". {}/etc/profile.d/conda.sh && conda activate simpy && ".format(_CONDA_PREFIX)
_POLICY_SERVER_YAML = lambda cluster_name, cloud_provider: "configs/{}_{}_policy_server.yaml".format(cluster_name,
                                                                                                     cloud_provider)

_POLICY_ACTOR_CONFIG = {'num_cpus': 1}

_CURRENT_ENV = sys.executable.split('/')[-3]


def _get_trainer_and_cloud(trainer_id: int):
    sql = "SELECT name, cloud_provider FROM trainer_cluster WHERE id = {}".format(P_MARKER)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id,))
    assert row is not None, "Unknown Trainer ID {}".format(trainer_id)
    return row
