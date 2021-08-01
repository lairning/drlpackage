import json
import subprocess
from datetime import datetime

from lairningdecisions.utils.db import _BACKOFFICE_DB, P_MARKER, select_record, SQLParamList
from lairningdecisions.utils.scaler_config import trainer_cluster_config, server_cluster_config
from lairningdecisions.utils.utils import _TRAINER_PATH, _TRAINER_YAML, _CMD_PREFIX, _SHELL, _get_trainer_and_cloud, \
    _POLICY_SERVER_YAML


# ToDo: P4 Add more exception handling
def launch_trainer(cluster_name: str = None, cloud_provider: str = '', cluster_config: dict = None,
                   template: str = 'simpy'):
    assert template in ['simpy', 'data'], "Invalid template '{}'".format(template)

    result = subprocess.run(['ls', _TRAINER_PATH(cluster_name, cloud_provider)], capture_output=True, text=True)
    # Create the Trainer Cluster if it does not exist.
    # No distinction exists between cloud providers, therefore training results are shared between runs in different
    # clouds
    if result.returncode != 0:
        # Create trainer folder
        template_path_dict = {'simpy': 'simpy_template', 'data': 'data_template'}
        result = subprocess.run(['cp', '-r', template_path_dict[template],
                                 _TRAINER_PATH(cluster_name, cloud_provider)],
                                capture_output=True,
                                text=True)
        if result.returncode:
            print("Error Creating Trainer Directory {}".format(_TRAINER_PATH(cluster_name, cloud_provider)))
            print(result.stderr)
            raise Exception('Cannot Create Trainer Directory')

        cursor = _BACKOFFICE_DB.cursor()
        sql = "INSERT INTO trainer_cluster (name, cloud_provider, last_start, status, config) VALUES ({})".format(
            SQLParamList(5))
        params = (cluster_name, cloud_provider, datetime.now(), 'up', json.dumps(cluster_config))
        cursor.execute(sql, params)
        trainer_id = cursor.lastrowid
    else:
        sql = '''SELECT id, last_start, status, config FROM trainer_cluster 
                 WHERE name = {} and cloud_provider = {}'''.format(P_MARKER, P_MARKER)
        row = select_record(_BACKOFFICE_DB, sql=sql, params=(cluster_name, cloud_provider))
        assert row, "{} exists without a record in the backoffice.db".format(
            _TRAINER_PATH(cluster_name, cloud_provider))
        trainer_id, last_start, status, config = select_record(_BACKOFFICE_DB, sql=sql,
                                                               params=(cluster_name, cloud_provider))

        if json.loads(config) != cluster_config or status == 'down':
            last_start = datetime.now()

        sql = '''UPDATE trainer_cluster SET last_start = {}, status = {}, config = {}
                      WHERE id = {}'''.format(P_MARKER, P_MARKER, P_MARKER, P_MARKER)
        params = (last_start, 'up', json.dumps(cluster_config), trainer_id)
        cursor = _BACKOFFICE_DB.cursor()
        cursor.execute(sql, params)

    _BACKOFFICE_DB.commit()

    # Create trainer yaml config file
    # When a cluster with the same name and provider is relaunched the configuration is overridden
    if cloud_provider != '':
        config_file = open(_TRAINER_YAML(cluster_name, cloud_provider), "wt")
        config_file.write(
            trainer_cluster_config(cloud_provider, cluster_name, _TRAINER_PATH(cluster_name, cloud_provider),
                                   config=cluster_config))
        config_file.close()
        # launch the cluster
        result = subprocess.run(_CMD_PREFIX + "ray up {} --no-config-cache -y".format(_TRAINER_YAML(
            cluster_name, cloud_provider)), shell=True, capture_output=True, text=True, executable=_SHELL)
        subprocess.run(_CMD_PREFIX + "ray exec {} 'rm -r /home/ubuntu/trainer/*'".format(
            _TRAINER_YAML(cluster_name, cloud_provider)),
                       shell=True, capture_output=True, text=True, executable=_SHELL)
        subprocess.run(_CMD_PREFIX + "ray rsync_up {} '{}/' '/home/ubuntu/trainer/'".format(
            _TRAINER_YAML(cluster_name, cloud_provider), _TRAINER_PATH(cluster_name, cloud_provider)),
                       shell=True, capture_output=True, text=True, executable=_SHELL)
    else:
        result = ''

    return trainer_id, result


# Get data from the cluster/trainer
def get_trainer_data(trainer_id: int):
    trainer_name, cloud_provider = _get_trainer_and_cloud(trainer_id=trainer_id)
    if cloud_provider != '':
        run_result = subprocess.run(_CMD_PREFIX + "ray rsync_down {} '/home/ubuntu/trainer/' '{}'".format(
            _TRAINER_YAML(trainer_name, cloud_provider), _TRAINER_PATH(trainer_name, cloud_provider)),
                                    shell=True, capture_output=True, text=True, executable=_SHELL)
        # Failed because the cluster is down
        if run_result.returncode == 1 and run_result.stderr.split('\n')[-2][
                                          :34] == 'RuntimeError: Head node of cluster':
            print("# {} is Down".format(_TRAINER_YAML(trainer_name, cloud_provider)))
            return False, ["ClusterDown"]
        # Failed for an unknown reason
        assert not run_result.returncode, "Error on SyncDown {} {}\n{}".format(
            _TRAINER_YAML(trainer_name, cloud_provider),
            _TRAINER_PATH(trainer_name, cloud_provider),
            run_result.stderr)
    # If trainer is 'local' or sync succeeded
    return True, []


def tear_down_trainer(trainer_id: int):
    sql = "SELECT name, cloud_provider FROM trainer_cluster WHERE id = {}".format(P_MARKER)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id,))
    assert row is not None, "Unknown Trainer ID {}".format(trainer_id)
    trainer_name, cloud_provider = row
    success, result = get_trainer_data(trainer_id=trainer_id)
    if success and cloud_provider != '':
        run_result = subprocess.run(_CMD_PREFIX + "ray down {} -y".format(_TRAINER_YAML(trainer_name, cloud_provider)),
                                    shell=True, capture_output=True, text=True, executable=_SHELL)
        assert not run_result.returncode, "Error on Tear Down {} \n{}".format(
            _TRAINER_YAML(trainer_name, cloud_provider),
            run_result.stderr)
    sql = '''UPDATE trainer_cluster SET status = {}
                  WHERE id = {}'''.format(P_MARKER, P_MARKER)
    params = ('down', trainer_id)
    cursor = _BACKOFFICE_DB.cursor()
    cursor.execute(sql, params)
    _BACKOFFICE_DB.commit()
    return success, result


def delete_trainer(trainer_id: int):
    sql = '''SELECT count(*) FROM policy 
             WHERE trainer_id = {} AND backend_name IS NOT NULL'''.format(P_MARKER, P_MARKER)
    count, = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id,))
    assert count == 0, "Can not delete trainer with deployed policies"
    tear_down_trainer(trainer_id=trainer_id)
    trainer_name, cloud_provider = _get_trainer_and_cloud(trainer_id=trainer_id)
    run_result = subprocess.run(['rm', '-r', _TRAINER_PATH(trainer_name, cloud_provider)], capture_output=True,
                                text=True)
    success, result = True, []
    if run_result.returncode:
        for line in run_result.stderr.split('\n'):
            print(line)
        success, result = False, ['RemoveTrainerPath']
    if cloud_provider != '':
        run_result = subprocess.run(['rm', _TRAINER_YAML(trainer_name, cloud_provider)], capture_output=True, text=True)
        if run_result.returncode:
            for line in run_result.stderr.split('\n'):
                print(line)
            success, result = False, result.append('RemoveTrainerConfig')
    cursor = _BACKOFFICE_DB.cursor()
    sql = '''DELETE FROM trainer_cluster WHERE id = {}'''.format(P_MARKER)
    cursor.execute(sql, (trainer_id,))
    _BACKOFFICE_DB.commit()
    return success, result


def launch_policy_server(cluster_name: str = None, cloud_provider: str = '', cluster_config: dict = None):
    policy_server_yaml = _POLICY_SERVER_YAML(cluster_name, cloud_provider)

    config_file = open(policy_server_yaml, "wt")

    config_file.write(server_cluster_config(cloud_provider, cluster_name, config=cluster_config))
    config_file.close()
    # launch the cluster
    run_result = subprocess.run(_CMD_PREFIX + "ray up {} --no-config-cache -y".format(policy_server_yaml),
                                shell=True, capture_output=True, text=True, executable=_SHELL)
    if run_result.returncode:
        return run_result  # False, ["RayUpFailed"]
    subprocess.run(_CMD_PREFIX + "ray exec {} 'rm -r /home/ubuntu/server/*'".format(policy_server_yaml),
                   shell=True, capture_output=True, text=True, executable=_SHELL)
    run_result = subprocess.run(_CMD_PREFIX + "ray rsync_up {} './' '/home/ubuntu/server/'".format(policy_server_yaml),
                                shell=True, capture_output=True, text=True, executable=_SHELL)
    if run_result.returncode:
        return run_result  # False, ["RaySyncFailed"]
    return run_result


def tear_down_policy_server(cluster_name: str = None, cloud_provider: str = ''):
    policy_server_yaml = _POLICY_SERVER_YAML(cluster_name, cloud_provider)

    success, result = True, []
    run_result = subprocess.run(_CMD_PREFIX + "ray down {} -y".format(policy_server_yaml),
                                shell=True, capture_output=True, text=True, executable=_SHELL)
    if run_result.returncode == 1 and run_result.stderr.split('\n')[-2][:34] == 'RuntimeError: Head node of cluster':
        print("# {} is Down".format(_TRAINER_YAML(cluster_name, cloud_provider)))
        success, result = False, ["ClusterAlreadyDown"]
    run_result = subprocess.run(['rm', policy_server_yaml], capture_output=True, text=True)
    if run_result.returncode == 1 and run_result.stderr.split('\n')[-2][:34] == 'RuntimeError: Head node of cluster':
        print("# Failed to remove {}".format(_TRAINER_YAML(cluster_name, cloud_provider)))
        success, result = False, result.append("UnknownYAML")
    return success, result


# ToDo: P3 not working properly
def monitor_policy_server(cluster_name: str = None, cloud_provider: str = ''):
    policy_server_yaml = _POLICY_SERVER_YAML(cluster_name, cloud_provider)
    result = ''
    if cloud_provider != '':
        result = subprocess.run(
            _CMD_PREFIX + "ray exec {} 'tail -n 18 -f /tmp/ray/session_latest/logs/monitor*'".format(
                policy_server_yaml), shell=True, capture_output=True, text=True, executable=_SHELL)
        assert not result.returncode, "Error on Monitoring {}\n{}".format(policy_server_yaml, result.stderr)
    return result


# ToDo: P3 not working properly
def monitor_trainer(trainer_id: int):
    trainer_name, cloud_provider = _get_trainer_and_cloud(trainer_id=trainer_id)
    result = ''
    if cloud_provider != '':
        result = subprocess.run(
            _CMD_PREFIX + "ray exec {} 'tail -n 18 /tmp/ray/session_latest/logs/monitor*'".format(
                _TRAINER_YAML(trainer_name, cloud_provider)), shell=True, capture_output=True, text=True,
            executable=_SHELL)
        # Failed because the cluster is down
        if result.returncode == 1 and result.stderr.split('\n')[-2][:34] == 'RuntimeError: Head node of cluster':
            print("# {} is Down".format(_TRAINER_YAML(trainer_name, cloud_provider)))
            return False, "ClusterDown"
        assert not result.returncode, "Error on Monitor {} \n{}".format(_TRAINER_YAML(trainer_name, cloud_provider),
                                                                        result.stderr)
    return result
