import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import ray
from lairningdecisions.utils.db import db_connect, TRAINER_DB_NAME, P_MARKER, select_record, SQLParamList, select_all
from ray.rllib.agents.dqn import DQNTrainer, ApexTrainer
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer, DDPPOTrainer
from ray.tune import run as tune_run
# from ray.tune import sample_from as tune_sample_from
from ray.tune.schedulers import PopulationBasedTraining

TRAINERS = {'ppo': PPOTrainer, 'appo': APPOTrainer, 'ddpo': DDPPOTrainer, 'dqn': DQNTrainer, 'apex_dqn': ApexTrainer}

from gym.spaces import Discrete
from gym import Env

"""Wrapper for a gym environment"""


class SimEnv(Env):

    def __init__(self, config: dict):
        self.action_space = Discrete(config["n_actions"])
        self.observation_space = config["observation_space"]
        self.sim_model = config["sim_model"]
        self.sim_config = config["sim_config"]

    def reset(self):
        self.sim = self.sim_model(config=self.sim_config)

        # Start processes and initialize resources
        obs = self.sim.get_observation()
        assert self.observation_space.contains(obs), "{} not in {}".format(obs, self.observation_space)
        return obs

    def step(self, action):
        assert action in range(self.action_space.n)

        self.sim.exec_action(action)
        obs = self.sim.get_observation()
        reward, done, info = self.sim.get_reward()

        assert self.observation_space.contains(obs), "{} not in {}".format(obs, self.observation_space)
        return obs, reward, done, info


class SimpyEnv(SimEnv):
    def __init__(self, config: dict):
        super().__init__(config)


class DataEnv(SimEnv):
    def __init__(self, config: dict):
        super().__init__(config)
        self.sim_data = config["sim_data"]

    def reset(self):
        self.sim = self.sim_model(config=self.sim_config, data=self.sim_data)

        # Start processes and initialize resources
        obs = self.sim.get_observation()
        assert self.observation_space.contains(obs), "{} not in {}".format(obs, self.observation_space)
        return obs


def cast_non_json(x):
    if isinstance(x, np.float32):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, dict):
        return {key: cast_non_json(value) for key, value in x.items()}
    return x


def filter_dict(dic_in: dict, keys: set):
    return {key: cast_non_json(dic_in[key]) for key in keys}


def my_ray_init():
    stderrout = sys.stderr
    sys.stderr = open('ray.log', 'w')
    try:
        ray.init(include_dashboard=False, log_to_driver=False, logging_level=0, address='auto')
    except ValueError:
        ray.init(include_dashboard=False, log_to_driver=False, logging_level=0)
    except Exception as e:
        raise e
    sys.stderr = stderrout


def get_available_cpus():
    return ray.available_resources().get('CPU', 0)


def my_ray_train(trainer):
    result = trainer.train()
    return result


class AISimAgent:
    default_config = {
        "num_cpus_per_worker": 1,
        # "batch_mode"         : "complete_episodes",
        "framework"          : "torch",
        "log_level"          : "ERROR",
    }

    # https://github.com/miyamotok0105/unity-ml-agents/blob/master/docs/Training-PPO.md
    #

    default_sim_config_name = "Base Config"
    default_sim_checkpoint_path = "./checkpoints"

    def __init__(self, sim_type: str, sim_name: str, log_level: str = "ERROR",
                 checkpoint_path=None, ai_config: dict = None, sim_config: dict = None,
                 csv_name: str = None, test_percent: float = 0.3):
        assert sim_type in ["DATA", "SIMPY"], 'sim_type must belong to ["DATA","SIMPY"]'
        self._sim_type = sim_type
        exec_locals = {}
        try:
            exec(
                "from data.{} import SimBaseline, N_ACTIONS, OBSERVATION_SPACE, SimModel, BASE_CONFIG".format(
                    sim_name), {}, exec_locals)
        except ModuleNotFoundError:
            raise Exception(" Model '{}' not found!!".format(sim_name))
        except Exception as e:
            raise e

        try:
            self.db = db_connect(TRAINER_DB_NAME)
        except Exception as e:
            raise e

        assert sim_config is None or isinstance(sim_config, dict), \
            "Parameter Simulation Config {} must be a dict!".format(sim_config)

        assert ai_config is None or isinstance(ai_config, dict), \
            "AI Agent Config {} must be a dict!".format(ai_config)

        assert isinstance(exec_locals['BASE_CONFIG'], dict), \
            "Default Simulation Config {} must be a dict!".format(exec_locals['BASE_CONFIG'])

        assert log_level in ["DEBUG", "INFO", "WARN", "ERROR"], "Invalid log_level {}".format(log_level)

        if not ray.is_initialized():
            my_ray_init()

        self._sim_baseline = exec_locals['SimBaseline']

        if sim_config is not None:
            exec_locals['BASE_CONFIG'].update(sim_config)
        sim_config = exec_locals['BASE_CONFIG']

        self._config = dict(self.default_config.copy())
        if ai_config is not None:
            self._config.update(ai_config)
        self._config["num_workers"] = int(get_available_cpus() - 1)
        self._config["log_level"] = log_level

        # ToDo: P2 Change the Observation Space to a function that receive a Sim Config as a parameter.
        #  In this part of the code it received exec_locals['BASE_CONFIG']
        self._config["env_config"] = {"n_actions"        : exec_locals['N_ACTIONS'],
                                      "observation_space": exec_locals['OBSERVATION_SPACE'],
                                      "sim_model"        : exec_locals['SimModel'],
                                      "sim_config"       : sim_config}

        if checkpoint_path is None:
            self.checkpoint_path = self.default_sim_checkpoint_path

        if self._sim_type == "DATA":
            exec_locals2 = {}
            try:
                exec(
                    "from data.{} import get_data".format(sim_name), {}, exec_locals2)
            except ModuleNotFoundError:
                raise Exception(" Model '{}' not found!!".format(sim_name))
            except Exception as e:
                raise e

            data = exec_locals2['get_data'](csv_name)
            self.training_data = data.iloc[:int(len(data) * (1 - test_percent))].copy()
            self.test_data = data.iloc[int(len(data) * (1 - test_percent)):].copy()

            # noinspection PyTypeChecker
            self._config["env"] = DataEnv
            # ToDo: P2 Change the Observation Space to a function that receive a Sim Config as a parameter.
            #  In this part of the code it received exec_locals['BASE_CONFIG']
            # noinspection PyTypeChecker
            self._config["env_config"]["sim_data"] = self.training_data
        else:
            self._config["env"] = SimpyEnv

        sql = '''SELECT id FROM sim_model WHERE name = {}'''.format(P_MARKER)
        params = (sim_name,)
        row = select_record(self.db, sql=sql, params=params)
        if row is None:
            cursor = self.db.cursor()
            cursor.execute('''INSERT INTO sim_model (name) VALUES ({})'''.format(P_MARKER), params)
            self._model_id = cursor.lastrowid
            params = (self._model_id, self.default_sim_config_name,
                      self._get_baseline_avg(sim_config), json.dumps(sim_config))
            cursor.execute('''INSERT INTO sim_config (sim_model_id,
                                                      name,
                                                      baseline_avg,
                                                      config) VALUES ({})'''.format(SQLParamList(4)), params)
            self.db.commit()
            print("# {} Created!".format(sim_name))
        else:
            self._model_id, = row

    def __del__(self):
        ray.shutdown()

    def _add_session(self, session_data: tuple):
        agent_type, agent_config, sim_config_id = session_data
        agent_config.pop("env", None)
        agent_config.pop("env_config", None)
        cursor = self.db.cursor()
        _session_data = (self._model_id, sim_config_id, datetime.now(), agent_type, json.dumps(agent_config))
        cursor.execute('''INSERT INTO training_session (
                                        sim_model_id,
                                        sim_config_id,
                                        time_start,
                                        agent_type,
                                        agent_config) VALUES ({})'''.format(SQLParamList(5)), _session_data)
        self.db.commit()
        return cursor.lastrowid

    def _get_sim_base_config(self):
        sql = '''SELECT id FROM sim_config 
                 WHERE sim_model_id = {} and name = {}'''.format(P_MARKER, P_MARKER)
        params = (self._model_id, self.default_sim_config_name)
        row = select_record(self.db, sql=sql, params=params)
        assert row is not None, "Base Sim Config not found!"
        return row[0]

    def _get_sim_config(self, sim_config: dict):
        cursor = self.db.cursor()
        if sim_config is None:
            sim_config_id = self._get_sim_base_config()
        else:
            sql = '''SELECT id, config FROM sim_config 
                     WHERE sim_model_id = {}'''.format(P_MARKER)
            params = (self._model_id,)
            row_list = select_all(self.db, sql=sql, params=params)
            # noinspection PyBroadException
            try:
                idx = [json.loads(config) for _, config in row_list].index(sim_config)
                sim_config_id, _ = row_list[idx]
            except Exception:
                params = (self._model_id, "Config {}".format(len(row_list)),
                          self._get_baseline_avg(sim_config), json.dumps(sim_config))
                cursor.execute('''INSERT INTO sim_config (sim_model_id,
                                                          name,
                                                          baseline_avg,
                                                          config) VALUES ({})'''.format(SQLParamList(4)), params)
                sim_config_id = cursor.lastrowid
        self.db.commit()
        return sim_config_id

    def _get_baseline_avg(self, sim_config: dict):
        @ray.remote
        def base_run(baseline):
            return baseline.run()

        if self._sim_type == "DATA":
            base = self._sim_baseline(sim_config=sim_config, data=self.training_data)
        else:
            base = self._sim_baseline(sim_config=sim_config)

        return np.mean(ray.get([base_run.remote(base) for _ in range(30)]))

    def _update_session(self, best_policy, duration, session_id):
        cursor = self.db.cursor()
        cursor.execute('''UPDATE training_session SET best_policy = {}, duration = {}
                          WHERE id = {}'''.format(P_MARKER, P_MARKER, P_MARKER), (best_policy, duration, session_id))
        self.db.commit()

    def _add_iteration(self, n, session_id, start_time, best_checkpoint, result):
        cursor = self.db.cursor()
        iteration_other_data_keys = {'info', 'training_iteration', 'experiment_id', 'date', 'timestamp',
                                     'time_this_iter_s'}
        iteration_data = (session_id, n, result['episode_reward_mean'], result['episode_reward_min'],
                          result['episode_reward_max'], best_checkpoint, (datetime.now() - start_time).total_seconds(),
                          start_time, json.dumps(filter_dict(result, iteration_other_data_keys)))
        cursor.execute('''INSERT INTO training_iteration (
                                        training_session_id,
                                        id,
                                        reward_mean,
                                        reward_min,
                                        reward_max,
                                        checkpoint,
                                        duration,
                                        time_start,
                                        other_data) VALUES ({})'''.format(SQLParamList(9)), iteration_data)
        self.db.commit()
        return cursor.lastrowid

    def _add_policy(self, policy_data: tuple):
        cursor = self.db.cursor()
        session_id, best_iteration, best_checkpoint, ai_type, agent_config, sim_config_id = policy_data
        agent_config.pop("env", None)
        agent_config.pop("env_config", None)
        agent_config = json.dumps(agent_config)
        policy_data = (self._model_id, sim_config_id, session_id, best_iteration, best_checkpoint, ai_type,
                       agent_config)
        cursor.execute('''INSERT INTO policy (
                                        sim_model_id,
                                        sim_config_id,
                                        session_id,
                                        iteration_id,
                                        checkpoint,
                                        agent_type,
                                        agent_config) VALUES ({})'''.format(SQLParamList(7)), policy_data)
        self.db.commit()
        return cursor.lastrowid

    def _add_policy_run(self, policy_run_data: tuple):
        cursor = self.db.cursor()
        cursor.execute('''INSERT INTO policy_run (
                                        policy_id,
                                        time_start,
                                        simulations,
                                        duration,
                                        results) VALUES ({})'''.format(SQLParamList(5)), policy_run_data)
        self.db.commit()
        return cursor.lastrowid

    def _add_baseline_run(self, policy_run_data: tuple):
        cursor = self.db.cursor()
        cursor.execute('''INSERT INTO baseline_run (
                                        sim_config_id,
                                        time_start,
                                        simulations,
                                        duration,
                                        results) VALUES ({})'''.format(SQLParamList(5)), policy_run_data)
        self.db.commit()
        return cursor.lastrowid

    # ToDo: P1 Retrain
    # Todo: Create tunning capabilities
    # ToDo: P3 Add more than one best policy
    def train(self, iterations: int = 10, ai_type: str = 'ppo', ai_config: dict = None, sim_config: dict = None,
              add_best_policy: bool = True, add_last_policy: bool = True, checkpoint_path=None):

        assert ai_type in TRAINERS.keys(), \
            'AI Type {} not Supported. Use one of the following {}'.format(ai_type, TRAINERS.keys())

        trainer_class = TRAINERS[ai_type]

        agent_config = self._config.copy()

        if ai_config is not None:
            assert isinstance(ai_config, dict), "Agent Config {} must be a dict!".format(ai_config)
            ai_config.pop("env", None)
            ai_config.pop("env_config", None)
            agent_config.update(ai_config)

        if sim_config is not None:
            assert isinstance(sim_config, dict), "Sim Config {} must be a dict!".format(sim_config)
            # ToDo: P2 Change the Observation Space to a function that receive a Sim Config as a parameter.
            #  In this part of the code the agent_config["env_config"]["observation_space"] have to be updated
            # noinspection PyTypeChecker
            agent_config["env_config"]["sim_config"].update(sim_config)

        if self._sim_type == "DATA":
            agent_config["env_config"]["sim_data"] = self.training_data

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        # noinspection PyTypeChecker
        sim_config_id = self._get_sim_config(agent_config["env_config"]["sim_config"])

        session_id = self._add_session((ai_type, agent_config.copy(), sim_config_id))

        print("# Training Session {} started at {}!".format(session_id, datetime.now()))

        trainer = trainer_class(config=agent_config)

        session_start = datetime.now()
        iteration_start = datetime.now()

        result = my_ray_train(trainer)

        best_checkpoint = trainer.save(checkpoint_dir=checkpoint_path)
        best_reward = result['episode_reward_mean']
        print("# Progress: {:2.1%} # Best Mean Reward: {:.2f}      ".format(1 / iterations, best_reward), end="\r")
        self._add_iteration(0, session_id, iteration_start, best_checkpoint, result)
        best_iteration = 0
        i = None

        for i in range(1, iterations):
            iteration_start = datetime.now()
            result = my_ray_train(trainer)
            # result = trainer.train()

            if result['episode_reward_mean'] > best_reward:
                best_checkpoint = trainer.save(checkpoint_dir=checkpoint_path)
                best_reward = result['episode_reward_mean']
                best_iteration = i
                checkpoint = best_checkpoint
            else:
                checkpoint = None
            print("# Progress: {:2.1%} # Best Mean Reward: {:.2f}      ".format((i + 1) / iterations, best_reward),
                  end="\r")
            self._add_iteration(i, session_id, iteration_start, checkpoint, result)

        print("# Progress: {:2.1%} # Best Mean Reward: {:.2f}      ".format(1, best_reward))
        self._update_session(best_iteration, (datetime.now() - session_start).total_seconds(), session_id)

        if add_best_policy:
            policy_data = (session_id, best_iteration, best_checkpoint, ai_type, agent_config.copy(), sim_config_id)
            self._add_policy(policy_data)

        if add_last_policy and i != best_iteration:
            checkpoint = trainer.save(checkpoint_dir=checkpoint_path)
            policy_data = (session_id, i, checkpoint, ai_type, agent_config.copy(), sim_config_id)
            self._add_policy(policy_data)

        print("# Training Session {} ended at {}!".format(session_id, datetime.now()))

    # ToDo: P0 install GPy
    def tune(self, experiment_name: str, num_samples: int = 8, iterations: int = 120, ai_type: str = 'ppo',
             ai_config: dict = None, search_space: dict = None, sim_config: dict = None, pbt_config: dict = None,
             checkpoint_path: str = None, verbose: int = 1) -> object:
        """

        :param verbose:
        :param pbt_config:
        :param experiment_name:
        :param num_samples:
        :param iterations:
        :param ai_type:
        :param ai_config:
        :param search_space: Parameters to search for the nest config.
            Example:   {"lambda": lambda: random.uniform(0.9, 1.0),
                        "clip_param": lambda: random.uniform(0.01, 0.5),
                        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5], }
        :param sim_config:
        :param checkpoint_path:
        """
        assert experiment_name is not None, "Tune needs a valid experiment_name"
        assert ai_type in TRAINERS.keys(), \
            'AI Type {} not Supported. Use one of the following {}'.format(ai_type, TRAINERS.keys())

        agent_config = self._config.copy()

        if ai_config is not None:
            assert isinstance(ai_config, dict), "Agent Config {} must be a dict!".format(ai_config)
            ai_config.pop("env", None)
            ai_config.pop("env_config", None)
            agent_config.update(ai_config)

        '''Code for pb2. It was decided to use PBT config_search_space = {key: tune_sample_from(lambda spec: fn(left, 
        right)) for key, (fn, left, right) in search_space.items()} agent_config.update(config_search_space) 

        hyperparam_bounds = {key: [left, right] for key, (fn, left, right) in search_space.items()}

        pb2 = PB2(time_attr = 'training_iteration', metric="episode_reward_mean", mode="max",
                  perturbation_interval=8, quantile_fraction=0.25, hyperparam_bounds=hyperparam_bounds)
        '''
        if pbt_config is None:
            pbt_config = dict()
        else:
            assert isinstance(pbt_config, dict), "pbt_config should be a dict"

        pbt = PopulationBasedTraining(time_attr=pbt_config.get('time_attr', 'training_iteration'),
                                      metric=pbt_config.get('metric', "episode_reward_mean"),
                                      mode=pbt_config.get('mode', "max"),
                                      perturbation_interval=pbt_config.get('perturbation_interval', 8),
                                      quantile_fraction=0.25, resample_probability=0.25,
                                      hyperparam_mutations=search_space)

        if sim_config is not None:
            assert isinstance(sim_config, dict), "Sim Config {} must be a dict!".format(sim_config)
            # ToDo: P2 Change the Observation Space to a function that receive a Sim Config as a parameter.
            #  In this part of the code the agent_config["env_config"]["observation_space"] have to be updated
            # noinspection PyTypeChecker
            agent_config["env_config"]["sim_config"].update(sim_config)

        if self._sim_type == "DATA":
            agent_config["env_config"]["sim_data"] = self.training_data

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        return tune_run(TRAINERS[ai_type], name=experiment_name, stop={"training_iteration": iterations},
                        num_samples=num_samples, local_dir=checkpoint_path, config=agent_config, scheduler=pbt,
                        verbose=verbose)

    def del_training_sessions(self, sessions: [int, list] = None):
        select_sessions_sql = '''SELECT id FROM training_session
                                 WHERE sim_model_id = {}'''.format(P_MARKER)
        params = (self._model_id,)
        all_sessions = select_all(self.db, sql=select_sessions_sql, params=params)
        all_sessions = {t[0] for t in all_sessions}
        del_sessions = []
        if isinstance(sessions, int):
            assert sessions in all_sessions, "Invalid session id {}".format(sessions)
            del_sessions = (sessions,)
        if isinstance(sessions, list):
            assert set(sessions).issubset(all_sessions), "Invalid sessions list {}".format(sessions)
            del_sessions = tuple(sessions)
        if sessions is None:
            del_sessions = tuple(all_sessions)
        if len(del_sessions):
            cursor = self.db.cursor()
            sql = '''DELETE FROM training_iteration
                     WHERE training_session_id IN ({})'''.format(SQLParamList(len(del_sessions)))
            cursor.execute(sql, del_sessions)
            sql = '''DELETE FROM training_session
                     WHERE id IN ({})'''.format(SQLParamList(len(del_sessions)))
            cursor.execute(sql, del_sessions)
            self.db.commit()

    def get_training_sessions(self):
        sql = '''SELECT training_session_id 
                 FROM training_iteration
                 WHERE sim_model_id = {}'''.format(P_MARKER)
        params = (self._model_id,)
        df = pd.read_sql_query(sql, self.db, params=params)
        return df

    def get_sim_config(self):
        sql = '''SELECT id, name, baseline_avg, config
                 FROM sim_config
                 WHERE sim_model_id = {}'''.format(P_MARKER)
        params = (self._model_id,)
        df = pd.read_sql_query(sql, self.db, params=params)
        return df

    # ToDo: P3 Change to allow more than one config
    def get_training_data(self, sim_config: int = None, baseline: bool = True):

        if sim_config is None:
            sim_config = self._get_sim_base_config()
        else:
            sql = "SELECT id FROM sim_config WHERE id = {}".format(P_MARKER)
            row = select_record(self.db, sql=sql, params=(sim_config,))
            assert row is not None, "Invalid Sim Config id {}".format(sim_config)
            sim_config, = row

        sql = '''SELECT training_session_id as session, training_iteration.id as iteration, reward_mean 
                 FROM training_iteration
                 INNER JOIN training_session ON training_iteration.training_session_id = training_session.id
                 INNER JOIN sim_config ON training_session.sim_config_id = sim_config.id
                 WHERE training_session.sim_config_id = {}'''.format(P_MARKER)
        params = (sim_config,)
        df = pd.read_sql_query(sql, self.db, params=params) \
            .pivot(index='iteration', columns='session', values='reward_mean')

        if baseline:
            sql = "SELECT baseline_avg FROM sim_config WHERE id = {}".format(P_MARKER)
            baseline_avg, = select_record(self.db, sql=sql, params=(sim_config,))
            df['baseline'] = [baseline_avg for _ in range(df.shape[0])]

        return df

    def get_policies(self):
        sql = '''SELECT agent_type, sim_config_id as sim_config, id as policy, session_id as session
                 FROM policy
                 WHERE sim_model_id = {}'''.format(P_MARKER)
        return pd.read_sql_query(sql, self.db, params=(self._model_id,))

    # ToDo P2: Optimize the evaluation of policies by using Ray Serve and multiple replicas
    def run_policies(self, policy: [int, list] = None, simulations: int = 1):

        select_policy_sql = '''SELECT id FROM policy
                                 WHERE sim_model_id = {}'''.format(P_MARKER)
        all_policies = select_all(self.db, sql=select_policy_sql, params=(self._model_id,))
        all_policies = {t[0] for t in all_policies}

        if isinstance(policy, int):
            assert policy in all_policies, "Invalid session id {}".format(policy)
            policies = (policy,)
        elif isinstance(policy, list):
            assert set(policy).issubset(all_policies), "Invalid sessions list {}".format(policy)
            policies = tuple(policy)
        else:
            policies = tuple(all_policies)

        select_policy_sql = '''SELECT policy.id, checkpoint, agent_type, agent_config, sim_config.config as s_config
                               FROM policy INNER JOIN sim_config ON policy.sim_config_id = sim_config.id
                               WHERE policy.id IN ({})'''.format(SQLParamList(len(policies)))
        policy_data = select_all(self.db, sql=select_policy_sql, params=policies)

        for policy_id, checkpoint, agent_type, saved_agent_config, saved_sim_config in policy_data:

            print("# Running AI Policy {} started at {}!".format(policy_id, datetime.now()))

            agent_config = self._config.copy()
            agent_config.update(json.loads(saved_agent_config))
            agent_config["num_workers"] = 0

            sim_config = json.loads(saved_sim_config)

            trainer_class = TRAINERS[agent_type]

            agent = trainer_class(config=agent_config)
            agent.restore(checkpoint)

            time_start = datetime.now()

            agent_config["env_config"]["sim_config"].update(sim_config)

            if self._sim_type == "DATA":
                agent_config["env_config"]["sim_data"] = self.test_data
                data_env = DataEnv(agent_config["env_config"])
            else:
                data_env = SimpyEnv(agent_config["env_config"])

            result_list = []
            for i in range(simulations):
                episode_reward = 0
                done = False
                obs = data_env.reset()
                while not done:
                    action = agent.compute_action(obs)
                    obs, reward, done, info = data_env.step(action)
                    episode_reward += reward
                result_list.append(episode_reward)
                print("# Progress: {:2.1%} ".format((i + 1) / simulations), end="\r")
            policy_run_data = (policy_id, time_start, simulations,
                               (datetime.now() - time_start).total_seconds(), json.dumps(result_list))
            self._add_policy_run(policy_run_data)
            print("# Progress: {:2.1%} ".format(1))
            print("# Running AI Policy {} ended at {}!".format(policy_id, datetime.now()))

    # Todo: P1 Add Baselines config
    def run_baselines(self, sim_config: [int, list] = None, simulations: int = 1):

        @ray.remote
        def base_run(base):
            return base.run()

        if sim_config is None:

            select_sim_sql = '''SELECT id, config FROM sim_config
                                WHERE sim_config.sim_model_id = {}'''.format(P_MARKER)
            rows = select_all(self.db, sql=select_sim_sql, params=(self._model_id,))
            sim_configs = ((i, json.loads(config)) for i, config in rows)
        else:
            if isinstance(sim_config, int):
                sim_config = [sim_config]
            if isinstance(sim_config, list):
                # Get all policies for the list of sim_configs
                select_sim_sql = '''SELECT id, config FROM sim_config
                                    WHERE id IN ({})'''.format(SQLParamList(len(sim_config)))
                rows = select_all(self.db, sql=select_sim_sql, params=tuple(sim_config))
                sim_configs = ((i, json.loads(config)) for i, config in rows)
            else:
                raise Exception("Invalid Sim Config {}".format(sim_config))

        for sim_config_id, sim_config in sim_configs:
            if self._sim_type == "DATA":
                baseline = self._sim_baseline(sim_config=sim_config, data=self.test_data)
            else:
                baseline = self._sim_baseline(sim_config=sim_config)

            print("# Baseline Simulation for Config {} started at {}!".format(sim_config_id, datetime.now()))
            time_start = datetime.now()
            result_list = ray.get([base_run.remote(baseline) for _ in range(simulations)])

            policy_run_data = (sim_config_id, time_start, simulations,
                               (datetime.now() - time_start).total_seconds(), json.dumps(result_list))
            self._add_baseline_run(policy_run_data)

            print("# Baseline Simulation for Config {} ended at {}!".format(sim_config_id, datetime.now()))

    def get_policy_run_data(self, sim_config: int = None, baseline: bool = True):

        if sim_config is None:
            sim_config = self._get_sim_base_config()
        else:
            sql = "SELECT id FROM sim_config WHERE id = {}".format(P_MARKER)
            row = select_record(self.db, sql=sql, params=(sim_config,))
            assert row is not None, "Invalid Sim Config id {}".format(sim_config)
            sim_config, = row

        sql = '''SELECT policy_id, policy_run.id, time_start, results 
                 FROM policy_run
                 INNER JOIN policy ON policy_run.policy_id = policy.id
                 WHERE policy.sim_config_id = {}'''.format(P_MARKER)
        params = (sim_config,)
        policy_run = select_all(self.db, sql=sql, params=params)
        df = pd.DataFrame([["ai_policy{}_run{}".format(policy_id, run_id), time, x]
                           for policy_id, run_id, time, l in policy_run for x in json.loads(l)],
                          columns=['policy', 'time', 'reward'])
        if baseline:
            sql = '''SELECT id, time_start, results 
                     FROM baseline_run
                     WHERE sim_config_id = {}'''.format(P_MARKER)
            params = (sim_config,)
            baseline_run = select_all(self.db, sql=sql, params=params)
            df2 = pd.DataFrame([["baseline_run{}".format(run_id), time, x]
                                for run_id, time, l in baseline_run for x in json.loads(l)],
                               columns=['policy', 'time', 'reward'])
            df = df.append(df2)

        return df

    def get_policy_statistics(self, sim_config: int = None, baseline: bool = True):
        df = self.get_policy_run_data(sim_config=sim_config, baseline=baseline)
        df2 = pd.DataFrame(columns=['policy', 'mean', 'std'])
        df2['policy'] = df.columns
        df2['mean'] = [df[col].mean() for col in df.columns]
        df2['std'] = [df[col].std() for col in df.columns]
        return df2


class AISimpyAgent(AISimAgent):
    def __init__(self, sim_name: str, log_level: str = "ERROR", checkpoint_path=None, ai_config: dict = None,
                 sim_config: dict = None):
        super().__init__("SIMPY", sim_name, log_level, checkpoint_path, ai_config, sim_config)


class AIDataAgent(AISimAgent):
    def __init__(self, sim_name: str, csv_name: str, test_percent: float = 0.3, log_level: str = "ERROR",
                 checkpoint_path=None, ai_config: dict = None, sim_config: dict = None):
        super().__init__("DATA", sim_name, log_level, checkpoint_path, ai_config, sim_config, csv_name, test_percent)


import simpy


class SimpyModel(simpy.Environment):
    def __init__(self, base_config: dict, config: dict = None):
        assert base_config is not None, "base_config cannot be None"
        super().__init__()
        self.sim_config = base_config.copy()
        if config is not None:
            self.sim_config.update(config)
        self.time = self.now
        assert {"SIM_DURATION", "ACTION_INTERVAL"} <= set(config.keys()), \
            '{"SIM_DURATION", "ACTION_INTERVAL"} not defined'
        self.sim_duration = config["SIM_DURATION"]
        self.step_time = config["ACTION_INTERVAL"]

    def run_until_action(self):
        self.run(until=self.time + self.step_time)
        self.time = self.now

    def done(self):
        return self.now >= self.sim_duration

    def exec_action(self, action):
        raise Exception("Not Implemented!!!")

    def get_observation(self):
        raise Exception("Not Implemented!!!")

    def get_reward(self):
        raise Exception("Not Implemented!!!")
