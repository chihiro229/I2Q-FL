import json
import os
import datetime
import hashlib
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from enum import Enum
from threading import RLock
from typing import (Any, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple,
                    Union)

import yaml

from .filename_checker import check_filename
from .role import Role
from .singleton import Singleton

RawConfigurationDict = Dict[str, Any]

DEFAULT_D_CFG_FILENAME_YAML = '1_data_config.yml'
DEFAULT_MDL_CFG_FILENAME_YAML = '2_model_config.yml'
DEFAULT_RT_CFG_FILENAME_YAML = '3_runtime_config.yml'

DEFAULT_D_CFG_FILENAME_JSON = '1_data_config.yml'
DEFAULT_MDL_CFG_FILENAME_JSON = '2_model_config.yml'
DEFAULT_RT_CFG_FILENAME_JSON = '3_runtime_config.yml'

# default data configurations
_D_BETA_DIRICHLET = 'beta_parameter'
_D_DIR_KEY = 'data_dir'
_D_NAME_KEY = 'dataset'
_D_NI_ENABLE_KEY = 'non-iid'
_D_NI_CLASS_KEY = 'non-iid-class'
_D_NI_STRATEGY_KEY = 'non-iid-strategy'
_D_NORMALIZE_KEY = 'normalize'
_D_SAMPLE_SIZE_KEY = 'sample_size'
_D_PARTITION_KEY = 'train_val_test'
_D_FEATURE_SIZE = 'feature_size'
_D_RANDOM_SEED = 'random_seed'
_DEFAULT_D_CFG: RawConfigurationDict = {
    _D_DIR_KEY: 'data',
    _D_NAME_KEY: 'mnist',
    _D_NI_ENABLE_KEY: False,
    _D_NI_CLASS_KEY: 1,
    _D_NI_STRATEGY_KEY: 'average',
    _D_NORMALIZE_KEY: True,
    _D_SAMPLE_SIZE_KEY: 300,
    _D_PARTITION_KEY: [0.8, 0.1, 0.1],
    _D_FEATURE_SIZE: 1000,
    _D_RANDOM_SEED: 100
}

# default model configurations
_STRATEGY_KEY = 'FedModel'
_STRATEGY_NAME_KEY = 'name'
_STRATEGY_ETA_KEY = 'eta'
_STRATEGY_B_KEY = 'B'
_STRATEGY_C_KEY = 'C'
_STRATEGY_E_KEY = 'E'
_STRATEGY_E_RATIO = 'evaluate_ratio'
_STRATEGY_E_DISTRIBUTE = 'distributed_evaluate'
_STRATEGY_MAX_ROUND_NUM_KEY = 'max_rounds'
_STRATEGY_TOLERANCE_NUM_KEY = 'num_tolerance'
_STRATEGY_NUM_ROUNDS_BETWEEN_VAL_KEY = 'rounds_between_val'
_STRATEGY_FEDSTC_SPARSITY_KEY = 'sparsity'
_STRATEGY_FEDPROX_MU_KEY = 'mu'
_STRATEGY_FEDOPT_TAU_KEY = 'tau'
_STRATEGY_FEDOPT_BETA1_KEY = 'beta1'
_STRATEGY_FEDOPT_BETA2_KEY = 'beta2'
_STRATEGY_FEDOPT_NAME_KEY = 'opt_name'
_STRATEGY_FETCHSGD_COL_NUM_KEY = 'num_col'
_STRATEGY_FETCHSGD_ROW_NUM_KEY = 'num_row'
_STRATEGY_FETCHSGD_BLOCK_NUM_KEY = 'num_block'
_STRATEGY_FETCHSGD_TOP_K_KEY = 'top_k'
_STRATEGY_FEDSVD_BLOCK = 'block_size'
_STRATEGY_FEDSVD_MODE = 'fedsvd_mode'
_STRATEGY_FEDSVD_TOPK = 'fedsvd_top_k'
_STRATEGY_FEDSVD_L2 = 'fedsvd_lr_l2'
_STRATEGY_FEDSVD_OPT_1 = 'fedsvd_opt_1'
_STRATEGY_FEDSVD_OPT_2 = 'fedsvd_opt_2'
_STRATEGY_FEDSVD_EVALUATE = 'fedsvd_debug_evaluate'

_ML_KEY = 'MLModel'
_ML_NAME_KEY = 'name'
_ML_ACTIVATION_KEY = 'activation'
_ML_DROPOUT_RATIO_KEY = 'dropout'
_ML_UNITS_SIZE_KEY = 'units'
_ML_OPTIMIZER_KEY = 'optimizer'
_ML_OPTIMIZER_NAME_KEY = 'name'
_ML_OPTIMIZER_LEARNING_RATE_KEY = 'lr'
_ML_OPTIMIZER_MOMENTUM_KEY = 'momentum'
_ML_LOSS_CALC_METHODS_KEY = 'loss'
_ML_METRICS_KEY = 'metrics'
_ML_THRESHOLD = 'threshold'
_ML_DEFAULT_METRICS = ['accuracy']
_DEFAULT_MDL_CFG: RawConfigurationDict = {
    _STRATEGY_KEY: {
        _STRATEGY_NAME_KEY: 'FedAvg',
        # shared params
        _STRATEGY_B_KEY: 32,
        _STRATEGY_C_KEY: 0.1,
        _STRATEGY_E_KEY: 1,
        _STRATEGY_E_RATIO: 1.0,
        _STRATEGY_E_DISTRIBUTE: True,
        _STRATEGY_MAX_ROUND_NUM_KEY: 3000,
        _STRATEGY_TOLERANCE_NUM_KEY: 100,
        _STRATEGY_NUM_ROUNDS_BETWEEN_VAL_KEY: 1,
        # FedSTC
        _STRATEGY_FEDSTC_SPARSITY_KEY: 0.01,
        # FedProx
        _STRATEGY_FEDPROX_MU_KEY: 0.01,
        # FedOpt
        _STRATEGY_FEDOPT_TAU_KEY: 1e-4,
        _STRATEGY_FEDOPT_BETA1_KEY: 0.9,
        _STRATEGY_FEDOPT_BETA2_KEY: 0.99,
        _STRATEGY_FEDOPT_NAME_KEY: 'fedyogi',
        # server-side learning rate, used by FedSCA and FedOpt
        _STRATEGY_ETA_KEY: 1.0,
        # FetchSGD
        _STRATEGY_FETCHSGD_COL_NUM_KEY: 5,
        _STRATEGY_FETCHSGD_ROW_NUM_KEY: 1e4,
        _STRATEGY_FETCHSGD_BLOCK_NUM_KEY: 10,
        _STRATEGY_FETCHSGD_TOP_K_KEY: 0.1,
    },
    _ML_KEY: {
        _ML_NAME_KEY: 'MLP',
        _ML_ACTIVATION_KEY: 'relu',
        _ML_DROPOUT_RATIO_KEY: 0.2,
        _ML_UNITS_SIZE_KEY: [512, 512],
        _ML_OPTIMIZER_KEY: {
            _ML_OPTIMIZER_NAME_KEY: 'sgd',
            _ML_OPTIMIZER_LEARNING_RATE_KEY: 0.1,
            _ML_OPTIMIZER_MOMENTUM_KEY: 0,
            # _ML_OPTIMIZER_MOMENTUM_KEY: 0.9,    # FetchSGD
        },
        _ML_LOSS_CALC_METHODS_KEY: 'categorical_crossentropy',
        _ML_METRICS_KEY: _ML_DEFAULT_METRICS,
    },
}

# default runtime configurations
# _RT_CLIENTS_KEY = 'clients'
# _RT_C_BANDWIDTH_KEY = 'bandwidth'

_RT_SERVER_KEY = 'server'
_RT_S_HOST_KEY = 'host'
_RT_S_LISTEN_KEY = 'listen'
_RT_S_PORT_KEY = 'port'
_RT_S_CLIENTS_NUM_KEY = 'num_clients'
_RT_S_SECRET_KEY = 'secret_key'

_RT_DOCKER_KEY = 'docker'
_RT_D_IMAGE_LABEL_KEY = 'image'
_RT_D_CONTAINER_NUM_KEY = 'num_containers'
_RT_D_GPU_ENABLE_KEY = 'enable_gpu'
_RT_D_GPU_NUM_KEY = 'num_gpu'

_RT_MACHINES_KEY = 'machines'
_RT_M_ADDRESS_KEY = 'host'
_RT_M_PORT_KEY = 'port'
_RT_M_USERNAME_KEY = 'username'
_RT_M_WORK_DIR_KEY = 'dir'
_RT_M_SK_FILENAME_KEY = 'key'
_RT_M_CAPACITY_KEY = 'capacity'
_RT_M_SERVER_NAME = 'server'

_RT_LOG_KEY = 'log'
_RT_L_BASE_LEVEL_KEY = 'base_level'
_RT_L_FILE_LEVEL_KEY = 'file_log_level'
_RT_L_CONSOLE_LEVEL_KEY = 'console_log_level'
_RT_L_DIR_PATH_KEY = 'log_dir'

_RT_COMMUNICATION_KEY = 'communication'
_RT_COMM_METHOD_KEY = 'method'
_RT_COMM_PORT_KEY = 'port'
_RT_COMM_LIMIT_FLAG_KEY = 'limit_network_resource'
_RT_COMM_BANDWIDTH_UP_KEY = 'bandwidth_upload'
_RT_COMM_BANDWIDTH_DOWN_KEY = 'bandwidth_download'
_RT_COMM_LATENCY_KEY = 'latency'
_RT_COMM_FAST_MODE = 'fast_mode'

_DEFAULT_RT_CFG: RawConfigurationDict = {
    _RT_COMMUNICATION_KEY: {
        _RT_COMM_METHOD_KEY: 'SocketIO',
        _RT_COMM_PORT_KEY: 8000,
        _RT_COMM_LIMIT_FLAG_KEY: True,
        _RT_COMM_BANDWIDTH_UP_KEY: '30Mbit',
        _RT_COMM_BANDWIDTH_DOWN_KEY: '10Mbit',
        _RT_COMM_LATENCY_KEY: '50ms',
        _RT_COMM_FAST_MODE: False
    },
    _RT_LOG_KEY: {
        _RT_L_DIR_PATH_KEY: 'log/quickstart',
        _RT_L_BASE_LEVEL_KEY: 'INFO',
        _RT_L_FILE_LEVEL_KEY: 'INFO',
        _RT_L_CONSOLE_LEVEL_KEY: 'ERROR',
    },
    _RT_DOCKER_KEY: {
        _RT_D_IMAGE_LABEL_KEY: 'fedeval:sdfsdf',
        _RT_D_CONTAINER_NUM_KEY: 10,
        _RT_D_GPU_ENABLE_KEY: False,
        _RT_D_GPU_NUM_KEY: 0,
    },
    # _RT_CLIENTS_KEY: {
    #     _RT_C_BANDWIDTH_KEY: '100Mbit',
    # },
    _RT_SERVER_KEY: {
        _RT_S_HOST_KEY: 'server',
        _RT_S_LISTEN_KEY: 'server',
        _RT_S_CLIENTS_NUM_KEY: 10,
        _RT_S_PORT_KEY: 8000,
        _RT_S_SECRET_KEY: 'secret!',
    },
    _RT_MACHINES_KEY: {
        _RT_M_SERVER_NAME: {
            _RT_M_ADDRESS_KEY: '10.173.1.22',
            _RT_M_PORT_KEY: 22,
            _RT_M_USERNAME_KEY: 'ubuntu',
            _RT_M_WORK_DIR_KEY: '/ldisk/chaidi/FedEval',
            _RT_M_SK_FILENAME_KEY: 'id_rsa',
        },
        'm1': {
            _RT_M_ADDRESS_KEY: '10.173.1.22',
            _RT_M_PORT_KEY: 22,
            _RT_M_USERNAME_KEY: 'ubuntu',
            _RT_M_WORK_DIR_KEY: '/ldisk/chaidi/FedEval',
            _RT_M_SK_FILENAME_KEY: 'id_rsa',

            _RT_M_CAPACITY_KEY: 100,
        },
    },
}


# --- Configuration Entities ---
class _Configuraiton(object):
    def __init__(self, config: RawConfigurationDict) -> None:
        self._inner: RawConfigurationDict = self._config_filter(config)

    @property
    def inner(self) -> RawConfigurationDict:
        """return a deep copy of its inner configuraiton data, presented as a dict.
        Noticed that modifications on the returned object will NOT affect the original
        configuration.

        Returns:
            RawConfigurationDict: a deep copy of the inner data representaiton
            of this config object.
        """
        return deepcopy(self._inner)

    @staticmethod
    def _config_filter(config: RawConfigurationDict) -> RawConfigurationDict:
        # No filter by default
        return config


class _DataConfig(_Configuraiton):
    _IID_EXCEPTiON_CONTENT = 'The dataset is configured as iid.'

    def __init__(self, data_config: RawConfigurationDict = _DEFAULT_D_CFG) -> None:
        super().__init__(data_config)

        # non-iid
        self._non_iid: bool = self._inner.get(_D_NI_ENABLE_KEY, False)
        if self._non_iid:
            self._non_iid_strategy_name: str = self._inner.get(
                _D_NI_STRATEGY_KEY, 'average')
            if self._non_iid_strategy_name != 'natural':
                self._non_iid_class_num: int = int(
                    self._inner.get(_D_NI_CLASS_KEY, 1))

        # partition
        partition = self._inner[_D_PARTITION_KEY].copy()
        if len(partition) != 3:
            raise ValueError(
                f'there should be 3 values in {_D_PARTITION_KEY}.')
        for i in partition:
            if i < 0:
                raise ValueError(
                    f'values in {_D_PARTITION_KEY} should not be negetive.')
        summation = sum(partition)
        if summation <= 1e-6:
            raise ValueError(f'values in {_D_PARTITION_KEY} are too small.')
        partition = [i / summation for i in partition]
        self._partition = partition

    @staticmethod
    def _config_filter(config: RawConfigurationDict) -> RawConfigurationDict:
        if not config[_D_NI_ENABLE_KEY]:
            config[_D_NI_CLASS_KEY] = None
            config[_D_NI_STRATEGY_KEY] = None
        return config

    @property
    def beta_value(self) -> float :
        """
        the concentration parameter beta for dirichlet distribution as per NIID-Bench
        Returns :
            int : the value of concentration parameter
        """
        return float(self._inner[_D_BETA_DIRICHLET])

    @property
    def dataset_name(self) -> str:
        """the name of the dataset, chosen from mnist, cifar10, cifar100, femnist, and mnist.

        Returns:
            str: the name of chosen dataset.
        """
        return self._inner[_D_NAME_KEY]

    @property
    def iid(self) -> bool:
        """if the dataset would be used in an i.i.d. manner.

        Returns:
            bool: True if the dataset is sampled in an i.i.d. manner; otherwise, False.
        """
        return not self._non_iid

    @property
    def non_iid_class_num(self) -> int:
        """return the number of classes hold by each client.
        Only avaliable when the dataset is sampled in a non-i.i.d. form.

        Raises:
            AttributeError: raised when called without non-i.i.d. setting.

        Returns:
            int: the number of classes hold by each client.
        """
        if self._non_iid:
            return self._non_iid_class_num
        else:
            raise AttributeError(_DataConfig._IID_EXCEPTiON_CONTENT)

    @property
    def non_iid_strategy_name(self) -> str:
        """return the name of non-i.i.d. data partition strategy.
        Two choices are given:
        1. "natural" strategy for femnist and celebA dataset
        2. "average" for mnist, cifar10 and cifar100

        Raises:
            AttributeError: raised when called without non-i.i.d. setting.

        Returns:
            str: the name of non-i.i.d. data partition strategy.
        """
        if self._non_iid:
            if not self._non_iid_strategy_name_check():
                raise AttributeError(
                    f'unregistered non-iid data partition strategy name: {self._non_iid_strategy_name}')
            return self._non_iid_strategy_name
        else:
            raise AttributeError(_DataConfig._IID_EXCEPTiON_CONTENT)

    def _non_iid_strategy_name_check(self) -> bool:
        """check if the non-i.i.d. data partition strategy is known.

        Returns:
            bool: True if the data partition strategy name is registered as followed; otherwise, False.
        """
        return self._non_iid_strategy_name in ['natural', 'average']

    @property
    def normalized(self) -> bool:
        """whether the image pixel data point will be normalized to [0, 1].

        Returns:
            bool: True if data points would be normalized; otherwise, False.
        """
        return self._inner[_D_NORMALIZE_KEY]

    @property
    def sample_size(self) -> int:
        """return the number of samples owned by each client."""
        if self._inner[_D_SAMPLE_SIZE_KEY] is None:
            return None
        return int(self._inner[_D_SAMPLE_SIZE_KEY])

    @property
    def data_partition(self) -> Sequence[float]:
        """get the data partition proportion, ordered as
        [train data ratio, test data ration, validation data ration].
        
        Constraints met by the return value:
            1. all the ratios in the returned list sum up to 1.
            2. all the ratios in the returned list are non-negative.

        Returns:
            Sequence[float]: [train data ratio, test data ration, validation data ration]
        """
        return self._partition

    @property
    def feature_size(self):
        # TODO(Di): Add constraints in the future
        # if self.dataset_name != 'synthetic_matrix_horizontal' and \
        #    self.dataset_name != 'synthetic_matrix_vertical':
        #     raise AttributeError
        return self._inner[_D_FEATURE_SIZE]

    @property
    def random_seed(self):
        return int(self._inner[_D_RANDOM_SEED])


class _ModelConfig(_Configuraiton):
    def __init__(self, model_config: RawConfigurationDict = _DEFAULT_MDL_CFG) -> None:
        _ModelConfig.__check_raw_config(model_config)
        super().__init__(model_config)
        self._strategy_cfg = model_config[_STRATEGY_KEY]
        # The model config could be empty, e.g., in FedSVD
        self._ml_cfg = model_config[_ML_KEY] or {}

    @staticmethod
    def _config_filter(config: RawConfigurationDict) -> RawConfigurationDict:
        # Fed Model filters
        if config[_STRATEGY_KEY][_STRATEGY_NAME_KEY] != 'FedSTC':
            config[_STRATEGY_KEY][_STRATEGY_FEDSTC_SPARSITY_KEY] = None
        if config[_STRATEGY_KEY][_STRATEGY_NAME_KEY] != 'FedProx':
            config[_STRATEGY_KEY][_STRATEGY_FEDPROX_MU_KEY] = None
        if config[_STRATEGY_KEY][_STRATEGY_NAME_KEY] != 'FedOpt':
            config[_STRATEGY_KEY][_STRATEGY_FEDOPT_TAU_KEY] = None
            config[_STRATEGY_KEY][_STRATEGY_FEDOPT_NAME_KEY] = None
            config[_STRATEGY_KEY][_STRATEGY_FEDOPT_BETA1_KEY] = None
            config[_STRATEGY_KEY][_STRATEGY_FEDOPT_BETA2_KEY] = None
        if config[_STRATEGY_KEY][_STRATEGY_NAME_KEY] != 'FedOpt' and \
                config[_STRATEGY_KEY][_STRATEGY_NAME_KEY] != 'FedSCA':
            config[_STRATEGY_KEY][_STRATEGY_ETA_KEY] = None
        return config

    @staticmethod
    def __check_raw_config(config: RawConfigurationDict) -> None:
        _ModelConfig.__check_runtime_config_shallow_structure(config)
        _ModelConfig.__check_ML_model_params(config.get(_ML_KEY))

    @staticmethod
    def __check_runtime_config_shallow_structure(config: RawConfigurationDict) -> None:
        # assert config.get(
        #     _ML_KEY) != None, f'model_config should have `{_ML_KEY}`'
        assert config.get(
            _STRATEGY_KEY) != None, f'model_config should have `{_STRATEGY_KEY}`'

    @staticmethod
    def __check_ML_model_params(ml_config: RawConfigurationDict) -> None:
        if ml_config:
            dropout_ratio = ml_config.get(_ML_DROPOUT_RATIO_KEY)
            if dropout_ratio:
                assert dropout_ratio >= 0 and dropout_ratio <= 1, 'dropout ration out of range.'

    @property
    def strategy_config(self) -> RawConfigurationDict:
        """a variant of inner method: return a copy of inner strategy raw dict.

        Returns:
            RawConfigurationDict: a deep copy of the strategy-related configuration dict.
        """
        return deepcopy(self._strategy_cfg)

    @property
    def ml_config(self) -> RawConfigurationDict:
        """a variant of inner method: return a copy of inner machine learning raw dict.

        Returns:
            RawConfigurationDict: a deep copy of the ML model-related configuration dict.
        """
        return deepcopy(self._ml_cfg)

    @property
    def strategy_name(self) -> str:
        """get the class name of the federated strategy (i.e., the main controller of federated
        process). Notice that the strategy class with this name (case sensitive and whole word
        matching) should have been implemented in this library (specifically, in strategy module),
        otherwise a TypeNotFound exception would be raised in the following steps.

        Returns:
            str: the classname/typename of the federated strategy.
        """
        return self._strategy_cfg[_STRATEGY_NAME_KEY]

    @property
    def ml_method_name(self) -> str:
        """get the class name of the machine learning model (i.e., the kernel of the whole
        calculation process). Notice that the strategy class with this name (case sensitive
        and whole word matching) should have been implemented in this library (specifically,
        in model module), otherwise a TypeNotFound exception would be raised in the
        following steps.

        Returns:
            str: the classname/typename of the inner machine learning model.
        """
        return self._ml_cfg.get(_ML_NAME_KEY)

    @property
    def server_learning_rate(self) -> float:
        """get the learning rate on the server side.
        Only available in FedOpt and FedSCA.

        Raises:
            AttributeError: called in a in proper federated strategy.

        Returns:
            float: the learning rate on the server side.
        """
        if self.strategy_name != 'FedOpt' and self.strategy_name != 'FedSCA':
            raise AttributeError
        return float(self._strategy_cfg[_STRATEGY_ETA_KEY])

    @property
    def B(self) -> int:
        """the local minibatch size used for the updates on the client side."""
        return int(self._strategy_cfg[_STRATEGY_B_KEY])

    @property
    def C(self) -> float:
        """the fraction of clients that perform computation in each round.

        Examples:
            if there are 100 available clients in a test network with a C of 0.2,
            then there should be (100*0.2=)20 clients in each round of iterations.
        """
        return float(self._strategy_cfg[_STRATEGY_C_KEY])

    @property
    def E(self) -> int:
        """the number of training passes that each client makes over its local dataset
        in each round.
        """
        return int(self._strategy_cfg[_STRATEGY_E_KEY])

    @property
    def evaluate_ratio(self):
        return float(self._strategy_cfg[_STRATEGY_E_RATIO])

    @property
    def distributed_evaluate(self):
        return bool(self._strategy_cfg[_STRATEGY_E_DISTRIBUTE])

    @property
    def max_round_num(self) -> int:
        """the total/maximum number of the iteration rounds."""
        return int(self._strategy_cfg[_STRATEGY_MAX_ROUND_NUM_KEY])

    @property
    def tolerance_num(self) -> int:
        """the patience for early stopping"""
        return int(self._strategy_cfg[_STRATEGY_TOLERANCE_NUM_KEY])

    @property
    def num_of_rounds_between_val(self) -> int:
        """the number of rounds between test or validation"""
        return int(self._strategy_cfg[_STRATEGY_NUM_ROUNDS_BETWEEN_VAL_KEY])

    @property
    def stc_sparsity(self) -> float:
        """TODO(fgh): the origin of FedSTC"""
        return float(self._strategy_cfg[_STRATEGY_FEDSTC_SPARSITY_KEY])

    @property
    def prox_mu(self) -> float:
        """the /mu parameter in FedProx, a scaler that measures the approximation
        between the local model and the global model.
        More info available in Federated Optimization in Heterogeneous Networks(arXiv:1812.06127).
        """
        return float(self._strategy_cfg[_STRATEGY_FEDPROX_MU_KEY])

    @property
    def opt_tau(self) -> float:
        # TODO(fgh) can not find a corresponding variable in FedOpt.
        return float(self._strategy_cfg[_STRATEGY_FEDOPT_TAU_KEY])

    @property
    def opt_beta_1(self) -> float:
        # TODO(fgh) can not find a corresponding variable in FedOpt.
        return float(self._strategy_cfg[_STRATEGY_FEDOPT_BETA1_KEY])

    @property
    def opt_beta_2(self) -> float:
        # TODO(fgh) can not find a corresponding variable in FedOpt.
        return float(self._strategy_cfg[_STRATEGY_FEDOPT_BETA2_KEY])

    @property
    def activation(self) -> str:
        """the name of activation mechanism in tensorflow layers.
        More info available in https://tensorflow.google.cn/api_docs/python/tf/keras/activations.
        """
        return self._ml_cfg[_ML_ACTIVATION_KEY]

    @property
    def dropout(self) -> float:
        """the dropout fraction of Dropout layer in the DL model."""
        return float(self._ml_cfg[_ML_DROPOUT_RATIO_KEY])

    @property
    def unit_size(self) -> Sequence[int]:
        """the size of sequential neural network components.

        Returns:
            Sequence[int]: the size of network components
            (ordered the same with data flow direction)
        """
        return [
            int(i) for i in self._ml_cfg[_ML_UNITS_SIZE_KEY]].copy()

    @property
    def optimizer_name(self) -> str:
        """the name of the optimizer in tensorflow network.
        More info available in https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers.
        """
        return self._ml_cfg[_ML_OPTIMIZER_KEY][_ML_OPTIMIZER_NAME_KEY]

    @property
    def learning_rate(self) -> float:
        """the learning rate of model training in tensorlflow."""
        return float(self._ml_cfg[_ML_OPTIMIZER_KEY][_ML_OPTIMIZER_LEARNING_RATE_KEY])

    @property
    def momentum(self) -> float:
        """the momentum of the optimizer."""
        return float(self._ml_cfg[_ML_OPTIMIZER_KEY][_ML_OPTIMIZER_MOMENTUM_KEY])

    @property
    def loss_calc_method(self) -> str:
        """the identifier of a loss function in tensorflow.
        More info available in https://tensorflow.google.cn/api_docs/python/tf/keras/losses.

        Returns:
            str: the string name of the loss function during model training.
        """
        return self._ml_cfg[_ML_LOSS_CALC_METHODS_KEY]

    @property
    def metrics(self) -> Sequence[str]:
        """names of the metrics used in model training and validation in tensorflow.
        More info in https://tensorflow.google.cn/api_docs/python/tf/keras/metrics.

        Returns:
            Sequence[str]: a copy of metric names.
        """
        return self._ml_cfg[_ML_METRICS_KEY].copy()

    @property
    def threshold(self) -> float:
        """threshold parameter to compare between rounds
        Returns :
            float : threshold value to compare the difference between accuraies between rounds
        """
        return float(self._ml_cfg[_ML_THRESHOLD])

    @property
    def col_num(self) -> int:
        """the number of columns in FetchSGD.
        More info available at https://export.arxiv.org/abs/2007.07682.
        """
        return int(self._ml_cfg[_STRATEGY_KEY][_STRATEGY_FETCHSGD_COL_NUM_KEY])

    @property
    def row_num(self) -> int:
        """the number of rows in FetchSGD.
        More info available at https://export.arxiv.org/abs/2007.07682.
        """
        return int(self._ml_cfg[_STRATEGY_KEY][_STRATEGY_FETCHSGD_ROW_NUM_KEY])

    @property
    def block_num(self) -> int:
        """the number of blocks in FetchSGD.
        More info available at https://export.arxiv.org/abs/2007.07682.
        """
        return int(self._ml_cfg[_STRATEGY_KEY][_STRATEGY_FETCHSGD_BLOCK_NUM_KEY])

    @property
    def top_k(self) -> int:
        """the number of top items in FetchSGD.
        More info available at https://export.arxiv.org/abs/2007.07682.
        """
        return int(self._ml_cfg[_STRATEGY_KEY][_STRATEGY_FETCHSGD_TOP_K_KEY])

    @property
    def block_size(self) -> int:
        """
        block size of FedSVD
        """
        if self.strategy_name != 'FedSVD':
            raise AttributeError
        return int(self._strategy_cfg[_STRATEGY_FEDSVD_BLOCK])

    @property
    def svd_mode(self) -> str:
        """
        block size of FedSVD
        """
        if self.strategy_name != 'FedSVD':
            raise AttributeError
        assert self._strategy_cfg[_STRATEGY_FEDSVD_MODE] in ['svd', 'pca', 'lr'], \
            f'Unknown FedSVD Mode: {self._strategy_cfg[_STRATEGY_FEDSVD_MODE]}, ' \
            f'should be either svd or pca'
        return str(self._strategy_cfg[_STRATEGY_FEDSVD_MODE])

    @property
    def svd_top_k(self) -> int:
        """
        block size of FedSVD
        """
        if self.strategy_name != 'FedSVD':
            raise AttributeError
        return int(self._strategy_cfg[_STRATEGY_FEDSVD_TOPK])

    @property
    def svd_lr_l2(self):
        """
        L2 penalize of FedSVD
        """
        if self.strategy_name != 'FedSVD':
            raise AttributeError
        return float(self._strategy_cfg[_STRATEGY_FEDSVD_L2])

    @property
    def svd_opt_1(self):
        if self.strategy_name != 'FedSVD':
            raise AttributeError
        return str(self._strategy_cfg[_STRATEGY_FEDSVD_OPT_1]).lower() == 'true'

    @property
    def svd_opt_2(self):
        if self.strategy_name != 'FedSVD':
            raise AttributeError
        return str(self._strategy_cfg[_STRATEGY_FEDSVD_OPT_2]).lower() == 'true'

    @property
    def svd_evaluate(self):
        if self.strategy_name != 'FedSVD':
            raise AttributeError
        return str(self._strategy_cfg[_STRATEGY_FEDSVD_EVALUATE]).lower() == 'true'


class _RT_Machine(_Configuraiton):
    __ITEM_CHECK_VALUE_ERROR_PATTERN = 'machine configuraitons should have {}.'

    def __init__(self, machine_config: RawConfigurationDict, is_server: bool = False) -> None:
        _RT_Machine.__check_items(machine_config, is_server)
        super().__init__(machine_config)
        self._is_server = is_server

    @staticmethod
    def __check_items(config: RawConfigurationDict, is_server: bool = False) -> None:
        required_keys = [_RT_M_ADDRESS_KEY, _RT_M_WORK_DIR_KEY,
                         _RT_M_PORT_KEY, _RT_M_USERNAME_KEY, _RT_M_SK_FILENAME_KEY]
        for k in required_keys:
            assert k in config, ValueError(
                _RT_Machine.__ITEM_CHECK_VALUE_ERROR_PATTERN.format(k))
        if not is_server:
            assert _RT_M_CAPACITY_KEY in config, ValueError(
                _RT_Machine.__ITEM_CHECK_VALUE_ERROR_PATTERN.format(_RT_M_CAPACITY_KEY))

    @property
    def is_server(self) -> bool:
        """if the machine is a central server."""
        return self._is_server

    @property
    def addr(self) -> str:
        """the IP address of this machine or the name of this container in docker."""
        return self._inner[_RT_M_ADDRESS_KEY]

    @property
    def port(self) -> int:
        """the port of this virtual machine on the physical machine."""
        return int(self._inner[_RT_M_PORT_KEY])

    @property
    def username(self) -> str:
        """the username of this machine."""
        return self._inner[_RT_M_USERNAME_KEY]

    @property
    def work_dir_path(self) -> str:
        """the path of this machine's working diretory."""
        return self._inner[_RT_M_WORK_DIR_KEY]

    @property
    def key_filename(self) -> str:
        """the name of ssh connection secret key file."""
        return self._inner[_RT_M_SK_FILENAME_KEY]

    @property
    def capacity(self) -> int:
        """the number of container that this machine can handle.
        Only available on the client side.

        Raises:
            AttributeError: called from the server side.
        """
        if self._is_server:
            raise AttributeError(
                'capacity is inaccessible for the server side.')
        return int(self._inner[_RT_M_CAPACITY_KEY])


class _RuntimeConfig(_Configuraiton):
    __ITEM_CHECK_VALUE_ERROR_PATTERN = 'runtime configurations should have {}.'
    __AVAILABLE_LOGGING_LEVELS = {
        'NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}

    def __init__(self, runtime_config: RawConfigurationDict = _DEFAULT_RT_CFG) -> None:
        _RuntimeConfig.__check_items(runtime_config)
        super().__init__(runtime_config)
        self.__init_machines()

    @staticmethod
    def _config_filter(config: RawConfigurationDict) -> RawConfigurationDict:
        if not config[_RT_COMMUNICATION_KEY][_RT_COMM_LIMIT_FLAG_KEY]:
            config[_RT_COMMUNICATION_KEY][_RT_COMM_BANDWIDTH_UP_KEY] = None
            config[_RT_COMMUNICATION_KEY][_RT_COMM_BANDWIDTH_DOWN_KEY] = None
            config[_RT_COMMUNICATION_KEY][_RT_COMM_LATENCY_KEY] = None
        return config

    @staticmethod
    def __check_items(config: RawConfigurationDict) -> None:
        required_keys = [_RT_DOCKER_KEY, _RT_SERVER_KEY,
                         _RT_COMMUNICATION_KEY, _RT_LOG_KEY]
        for k in required_keys:
            assert k in config, ValueError(
                _RuntimeConfig.__ITEM_CHECK_VALUE_ERROR_PATTERN.format(k))

    def _has_machines(self) -> bool:
        return _RT_MACHINES_KEY in self._inner

    def __init_machines(self) -> bool:
        if not self._has_machines():
            return False
        self._machines: Dict[str, _RT_Machine] = dict()
        for name in self._inner[_RT_MACHINES_KEY]:
            self._machines[name] = _RT_Machine(
                self._inner[_RT_MACHINES_KEY][name], name == _RT_M_SERVER_NAME)
        return True

    @property
    def machines(self) -> Optional[Mapping[str, _RT_Machine]]:
        """return a deep copy of all the machines in the configuration.

        Returns:
            Optional[Mapping[str, _RT_Machine]]: None if there is no machine setting.
        """
        if not self._has_machines():
            return None
        return deepcopy(self._machines)

    @property
    def client_machines(self) -> Optional[Mapping[str, _RT_Machine]]:
        """return a deep copy of all the client machines in the configuration.

        Returns:
            Optional[Mapping[str, _RT_Machine]]: None if there is no client machine setting.
        """
        if not self._has_machines():
            return None
        return deepcopy({name: v for name, v in self._machines.items() if not v.is_server})

    @property
    def server_machine(self):
        if not self._has_machines():
            return None
        server = [v for _, v in self._machines.items() if v.is_server]
        assert len(server) == 1, 'The system requires one server'
        return deepcopy(server[0])

    @property
    def limit_network_resource(self) -> bool:
        """whether limit the network resource"""
        return bool(self._inner[_RT_COMMUNICATION_KEY][_RT_COMM_LIMIT_FLAG_KEY])

    @property
    def bandwidth_upload(self) -> str:
        """the bandwidth of each container."""
        if not self.limit_network_resource:
            raise AttributeError
        return self._inner[_RT_COMMUNICATION_KEY][_RT_COMM_BANDWIDTH_UP_KEY]

    @property
    def bandwidth_download(self) -> str:
        """the bandwidth of each container."""
        if not self.limit_network_resource:
            raise AttributeError
        return self._inner[_RT_COMMUNICATION_KEY][_RT_COMM_BANDWIDTH_DOWN_KEY]

    @property
    def latency(self) -> str:
        """the latency of each container."""
        if not self.limit_network_resource:
            raise AttributeError
        return self._inner[_RT_COMMUNICATION_KEY][_RT_COMM_LATENCY_KEY]

    @property
    def image_label(self) -> str:
        """the label of the docker image used in this experiment."""
        return self._inner[_RT_DOCKER_KEY][_RT_D_IMAGE_LABEL_KEY]

    @property
    def container_num(self) -> int:
        """the number of total docker containers in this experiment."""
        return int(self._inner[_RT_DOCKER_KEY][_RT_D_CONTAINER_NUM_KEY])

    @property
    def central_server_addr(self) -> str:
        """the IP address of the central server."""
        return self._inner[_RT_SERVER_KEY][_RT_S_HOST_KEY]

    @property
    def central_server_listen_at(self) -> str:
        """the listening IP address of the flask services on the cetral server side."""
        return self._inner[_RT_SERVER_KEY][_RT_S_LISTEN_KEY]

    @property
    def central_server_port(self) -> int:
        """the port that the central server occupies."""
        return int(self._inner[_RT_SERVER_KEY][_RT_S_PORT_KEY])

    @property
    def client_num(self) -> int:
        """the total number of the clients."""
        return int(self._inner[_RT_SERVER_KEY][_RT_S_CLIENTS_NUM_KEY])

    @staticmethod
    def _check_log_level_validity(level: str) -> None:
        """make sure the given string is one of the logging levels.

        Args:
            level (str): a string representation of a logging level.

        Raises:
            ValueError: the given string is not a valid logging level.
        """
        if level not in _RuntimeConfig.__AVAILABLE_LOGGING_LEVELS:
            raise ValueError(
                f'invalid logging level, available choices: {_RuntimeConfig.__AVAILABLE_LOGGING_LEVELS}')

    @property
    def base_log_level(self) -> str:
        """the base logging level of all the loggers."""
        lvl = self._inner[_RT_LOG_KEY][_RT_L_BASE_LEVEL_KEY]
        _RuntimeConfig._check_log_level_validity(lvl)
        return lvl

    @property
    def file_log_level(self) -> str:
        """the logging level in the log files."""
        lvl = self._inner[_RT_LOG_KEY][_RT_L_FILE_LEVEL_KEY]
        _RuntimeConfig._check_log_level_validity(lvl)
        return lvl

    @property
    def console_log_level(self) -> str:
        """the logging level in consoles."""
        lvl = self._inner[_RT_LOG_KEY][_RT_L_CONSOLE_LEVEL_KEY]
        _RuntimeConfig._check_log_level_validity(lvl)
        return lvl

    @property
    def secret_key(self) -> str:
        """the secret key of the flask service on the central server side.

        Returns:
            str: the secret key as a string.
        """
        return self._inner[_RT_SERVER_KEY][_RT_S_SECRET_KEY]

    @property
    def gpu_enabled(self) -> bool:
        """whether the GPU is enabled in this experiment."""
        return bool(self._inner[_RT_DOCKER_KEY][_RT_D_GPU_ENABLE_KEY])

    @property
    def gpu_num(self) -> int:
        """the number of GPUs.

        Raises:
            AttributeError: called without GPUs enabled.
        """
        if not self.gpu_enabled:
            raise AttributeError('GPU is not enabled.')
        return int(self._inner[_RT_DOCKER_KEY][_RT_D_GPU_NUM_KEY])

    @property
    def comm_method(self) -> str:
        """the method/technique used for mechaine-wise communication in the experiment."""
        return self._inner[_RT_COMMUNICATION_KEY][_RT_COMM_METHOD_KEY]

    @property
    def comm_port(self) -> int:
        """the port for communication on the server side."""
        return int(self._inner[_RT_COMMUNICATION_KEY][_RT_COMM_PORT_KEY])

    @property
    def comm_fast_mode(self) -> bool:
        """
        In fast mode, all the clients in one container will only download the parameters once
         to improve the efficiency, e.g., when tuning the parameters.
        Turn off the fast_mode if you are benchmarking the communication and time
        """
        return bool(self._inner[_RT_COMMUNICATION_KEY][_RT_COMM_FAST_MODE])


# --- Configuration Manager Interfaces ---
class ConfigurationManagerInterface(ABC):
    @abstractproperty
    def data_config_filename(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def model_config_filename(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def runtime_config_filename(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def data_config(self) -> RawConfigurationDict:
        raise NotImplementedError

    @abstractproperty
    def model_config(self) -> _ModelConfig:
        raise NotImplementedError

    @abstractproperty
    def runtime_config(self) -> RawConfigurationDict:
        raise NotImplementedError

    @abstractproperty
    def job_id(self):
        raise NotImplementedError


class ClientConfigurationManagerInterface(ABC):
    """an interface of ConfigurationManager from the client side,
    regulating the essential functions as clients.

    Raises:
        NotImplementedError: called without implementation.
    """
    pass


class ServerConfigurationManagerInterface(ABC):
    """an interface of ConfigurationManager from the central server side,
    regulating the essential functions as clients.

    Raises:
        NotImplementedError: called without implementation.
    """

    @abstractproperty
    def num_of_train_clients_contacted_per_round(self) -> int:
        raise NotImplementedError


# --- Configuration Serilizer Interfaces ---
_DEFAULT_ENCODING = 'utf-8'
_Stream = Union[str, bytes, TextIO]


class _CfgYamlInterface(ABC):
    """an interface that regulates the methods used to serialize
    and deserialize configuraitons in YAML.
    """

    @staticmethod
    def load_configs(
            src_path,
            data_config_filename: str = DEFAULT_D_CFG_FILENAME_YAML,
            model_config_filename: str = DEFAULT_MDL_CFG_FILENAME_YAML,
            runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME_YAML,
            encoding=_DEFAULT_ENCODING
    ) -> Tuple[RawConfigurationDict, RawConfigurationDict, RawConfigurationDict]:
        _d_cfg_path = os.path.join(src_path, data_config_filename)
        _mdl_cfg_path = os.path.join(src_path, model_config_filename)
        _rt_cfg_path = os.path.join(src_path, runtime_config_filename)
        with open(_d_cfg_path, 'r', encoding=encoding) as f:
            d_cfg = yaml.safe_load(f)
        with open(_mdl_cfg_path, 'r', encoding=encoding) as f:
            mdl_cfg = yaml.safe_load(f)
        with open(_rt_cfg_path, 'r', encoding=encoding) as f:
            rt_cfg = yaml.safe_load(f)
        return d_cfg, mdl_cfg, rt_cfg

    @staticmethod
    def save_configs(
            data_cfg: RawConfigurationDict,
            model_cfg: RawConfigurationDict,
            runtime_cfg: RawConfigurationDict,
            dst_path,
            data_config_filename: str = DEFAULT_D_CFG_FILENAME_YAML,
            model_config_filename: str = DEFAULT_MDL_CFG_FILENAME_YAML,
            runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME_YAML,
            encoding=_DEFAULT_ENCODING
    ) -> None:
        os.makedirs(dst_path, exist_ok=True)
        _d_cfg_path = os.path.join(dst_path, data_config_filename)
        _mdl_cfg_path = os.path.join(dst_path, model_config_filename)
        _rt_cfg_path = os.path.join(dst_path, runtime_config_filename)
        with open(_d_cfg_path, 'w', encoding=encoding) as f:
            yaml.dump(data_cfg, f)
        with open(_mdl_cfg_path, 'w', encoding=encoding) as f:
            yaml.dump(model_cfg, f)
        with open(_rt_cfg_path, 'w', encoding=encoding) as f:
            yaml.dump(runtime_cfg, f)


class _CfgJsonInterface(ABC):
    """an interface that regulates the methods used to serialize
    and deserialize configuraitons in JSON.
    """

    @staticmethod
    def load_configs(
            src_path,
            data_config_filename: str = DEFAULT_D_CFG_FILENAME_JSON,
            model_config_filename: str = DEFAULT_MDL_CFG_FILENAME_JSON,
            runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME_JSON,
            encoding=_DEFAULT_ENCODING
    ) -> Tuple[RawConfigurationDict, RawConfigurationDict, RawConfigurationDict]:
        _d_cfg_path = os.path.join(src_path, data_config_filename)
        _mdl_cfg_path = os.path.join(src_path, model_config_filename)
        _rt_cfg_path = os.path.join(src_path, runtime_config_filename)
        with open(_d_cfg_path, 'r', encoding=encoding) as f:
            d_cfg = json.load(f)
        with open(_mdl_cfg_path, 'r', encoding=encoding) as f:
            mdl_cfg = json.load(f)
        with open(_rt_cfg_path, 'r', encoding=encoding) as f:
            rt_cfg = json.load(f)
        return d_cfg, mdl_cfg, rt_cfg

    @staticmethod
    def save_configs(
            data_cfg: RawConfigurationDict,
            model_cfg: RawConfigurationDict,
            runtime_cfg: RawConfigurationDict,
            dst_path,
            data_config_filename: str = DEFAULT_D_CFG_FILENAME_YAML,
            model_config_filename: str = DEFAULT_MDL_CFG_FILENAME_YAML,
            runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME_YAML,
            encoding=_DEFAULT_ENCODING
    ) -> None:
        os.makedirs(dst_path, exist_ok=True)
        _d_cfg_path = os.path.join(dst_path, data_config_filename)
        _mdl_cfg_path = os.path.join(dst_path, model_config_filename)
        _rt_cfg_path = os.path.join(dst_path, runtime_config_filename)
        with open(_d_cfg_path, 'w', encoding=encoding) as f:
            json.dump(data_cfg, f)
        with open(_mdl_cfg_path, 'w', encoding=encoding) as f:
            json.dump(model_cfg, f)
        with open(_rt_cfg_path, 'w', encoding=encoding) as f:
            json.dump(runtime_cfg, f)


class _CfgSerializer(Enum):
    """types of serializer for configurations."""
    YAML = 'yaml'
    JSON = 'json'


class _CfgFileInterface(ABC):
    """an interface that regulates the methods used to serialize
    and deserialize configuraitons from the file system.
    """

    @staticmethod
    @abstractmethod
    def from_files(from_config_path: str,
                   serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
                   encoding=_DEFAULT_ENCODING) -> ConfigurationManagerInterface:
        raise NotImplementedError

    @abstractmethod
    def to_files(self,
                 dst_dir_path: str,
                 serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
                 encoding: Optional[str] = None) -> None:
        raise NotImplementedError

    @staticmethod
    def serializer2enum(serializer: Union[str, _CfgSerializer]) -> _CfgSerializer:
        """convert serializer name(string) into enum type"""
        if isinstance(serializer, str):
            try:
                serializer = _CfgSerializer(serializer)
            except ValueError:
                raise ValueError(f'{serializer} is not supported currently.')
        if not isinstance(serializer, _CfgSerializer):
            raise ValueError(f'invalid serializer type: {serializer.__class__.__name__}.')
        return serializer


# --- Role-related Configuration Interface ---
class _RoledConfigurationInterface(ABC):
    @abstractproperty
    def role(self) -> Role:
        raise NotImplementedError


# --- Configuration Manager ---
class ConfigurationManager(Singleton,
                           ConfigurationManagerInterface,
                           ClientConfigurationManagerInterface,
                           ServerConfigurationManagerInterface,
                           _CfgYamlInterface,
                           _CfgJsonInterface,
                           _CfgFileInterface,
                           _RoledConfigurationInterface):
    __init_once_lock = RLock()  # thread lock for __initiated
    __initiated = False  # whether this class has been initiated

    def __init__(self,
                 data_config: RawConfigurationDict = _DEFAULT_D_CFG,
                 model_config: RawConfigurationDict = _DEFAULT_MDL_CFG,
                 runtime_config: RawConfigurationDict = _DEFAULT_RT_CFG,
                 thread_safe: bool = False) -> None:
        with ConfigurationManager.__init_once_lock:
            if not ConfigurationManager.__initiated:
                super().__init__(thread_safe)
                self._d_cfg: _DataConfig = _DataConfig(data_config)
                self._mdl_cfg: _ModelConfig = _ModelConfig(model_config)
                self._rt_cfg: _RuntimeConfig = _RuntimeConfig(runtime_config)

                self._job_time = os.environ.get('UNIFIED_JOB_TIME', datetime.datetime.now().strftime('%Y_%m%d_%H%M%S'))

                self._init_file_names()
                self._encoding = _DEFAULT_ENCODING
                self.__init_role()

                # added variable to set incremental amount of clients in each round as per new client selection strategy
                self._num_of_train_clients_contacted_per_round = 0

                # Set random seeds
                import tensorflow as tf
                import numpy as np
                tf.random.set_seed(self._d_cfg.random_seed)
                np.random.seed(self._d_cfg.random_seed)
                ConfigurationManager.__initiated = True

    def _init_file_names(self,
                         data_config_filename: str = DEFAULT_D_CFG_FILENAME_YAML,
                         model_config_filename: str = DEFAULT_MDL_CFG_FILENAME_YAML,
                         runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME_YAML) -> None:
        self._d_cfg_filename = data_config_filename
        self._mdl_cfg_filename = model_config_filename
        self._rt_cfg_filename = runtime_config_filename
        # TODO(fgh) add unit tests for this method in test_config.py

    @property
    def data_unique_id(self):
        # Collect the configs that determine the datasets,
        #   and generate a unique ID
        data_unique_configs = [f'{key}={value}' for key, value in self._d_cfg.inner.items()]
        data_unique_configs += [
            f'ml_model={self._mdl_cfg.ml_method_name}',
            f'n_clients={self._rt_cfg.client_num}',
        ]
        data_unique_configs = sorted(data_unique_configs)
        return self._get_md5(','.join(data_unique_configs))

    @property
    def config_unique_id(self):
        return self.generate_unique_id(self._d_cfg.inner, self._mdl_cfg.inner, self._rt_cfg.inner)

    @classmethod
    def generate_unique_id(cls, data_config: dict, model_config: dict, runtime_config: dict):
        unique_configs = [
            json.dumps(data_config, sort_keys=True),
            json.dumps(model_config, sort_keys=True),
            json.dumps(runtime_config, sort_keys=True)
        ]
        return cls._get_md5(','.join(unique_configs))

    @staticmethod
    def _get_md5(config_string):
        # Creat the hash code
        hl = hashlib.md5()
        hl.update(config_string.encode(encoding='utf-8'))
        return hl.hexdigest()

    @property
    def data_dir_name(self) -> str:
        """The output directory of the clients' data.

        Returns:
            str: the name of the data directory.
        """
        return os.path.join(self._d_cfg.inner[_D_DIR_KEY], f'{self._d_cfg.dataset_name}_{self.data_unique_id[:10]}')

    @property
    def log_dir_path(self) -> str:
        """the path of the base of log directory."""
        return os.path.join(
            self._rt_cfg.inner[_RT_LOG_KEY][_RT_L_DIR_PATH_KEY],
            self._job_time + '_' + self.config_unique_id
        )

    @property
    def history_record_path(self) -> str:
        """the path of the history record."""
        return self._rt_cfg.inner[_RT_LOG_KEY][_RT_L_DIR_PATH_KEY]

    @property
    @Singleton.thread_safe_ensurance
    def job_id(self) -> str:
        return str(self._job_time)

    @property
    @Singleton.thread_safe_ensurance
    def encoding(self) -> str:
        """the encoding scheme during (de)serialization."""
        return self._encoding

    @encoding.setter
    @Singleton.thread_safe_ensurance
    def encoding(self, encoding):
        self._encoding = encoding

    @property
    @Singleton.thread_safe_ensurance
    def data_config_filename(self) -> str:
        return self._d_cfg_filename

    @data_config_filename.setter
    @Singleton.thread_safe_ensurance
    @check_filename(1)
    def data_config_filename(self, filename: str):
        self._d_cfg_filename = filename

    @property
    @Singleton.thread_safe_ensurance
    def model_config_filename(self) -> str:
        return self._mdl_cfg_filename

    @model_config_filename.setter
    @Singleton.thread_safe_ensurance
    @check_filename(1)
    def model_config_filename(self, filename: str) -> None:
        self._mdl_cfg_filename = filename

    @property
    @Singleton.thread_safe_ensurance
    def runtime_config_filename(self) -> str:
        return self._rt_cfg_filename

    @runtime_config_filename.setter
    @Singleton.thread_safe_ensurance
    @check_filename(1)
    def runtime_config_filename(self, filename: str) -> None:
        self._rt_cfg_filename = filename

    @property
    def data_config(self) -> _DataConfig:
        return self._d_cfg

    @property
    def model_config(self) -> _ModelConfig:
        return self._mdl_cfg

    @property
    def runtime_config(self) -> _RuntimeConfig:
        return self._rt_cfg

    @property
    def num_of_train_clients_contacted_per_round(self) -> int:
        """the number of clients selected to participate the main
        federated process in each round.
        """
        self._num_of_train_clients_contacted_per_round = max(1, int(self._rt_cfg.client_num * self._mdl_cfg.C))
        return self._num_of_train_clients_contacted_per_round

    @num_of_train_clients_contacted_per_round.setter
    def num_of_train_clients_contacted_per_round(self, value) :
        """
        sets the number of train clients selected per round to allow variable clients per round
        """
        self._num_of_train_clients_contacted_per_round = value

    @property
    def num_of_eval_clients_contacted_per_round(self) -> int:
        """the number of clients selected to participate the main
        federated process in each round.
        """
        return max(1, int(self._rt_cfg.client_num * self._mdl_cfg.evaluate_ratio))

    @staticmethod
    def load_configs(
        src_path,
        serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
        data_config_filename: str = DEFAULT_D_CFG_FILENAME_YAML,
        model_config_filename: str = DEFAULT_MDL_CFG_FILENAME_YAML,
        runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME_YAML,
        encoding=_DEFAULT_ENCODING
    ) -> Tuple[RawConfigurationDict, RawConfigurationDict, RawConfigurationDict]:
        serializer = _CfgFileInterface.serializer2enum(serializer)
        if serializer == _CfgSerializer.YAML:
            return _CfgYamlInterface.load_configs(
                src_path, encoding=encoding,
                data_config_filename=data_config_filename,
                model_config_filename=model_config_filename,
                runtime_config_filename=runtime_config_filename
            )
        elif serializer == _CfgSerializer.JSON:
            return _CfgYamlInterface.load_configs(
                src_path, encoding=encoding,
                data_config_filename=data_config_filename,
                model_config_filename=model_config_filename,
                runtime_config_filename=runtime_config_filename
            )
        else:
            raise NotImplementedError(f'Invalid serializer {serializer}')

    @staticmethod
    def save_configs(
        data_cfg: RawConfigurationDict,
        model_cfg: RawConfigurationDict,
        runtime_cfg: RawConfigurationDict,
        dst_path,
        data_config_filename: str = DEFAULT_D_CFG_FILENAME_YAML,
        model_config_filename: str = DEFAULT_MDL_CFG_FILENAME_YAML,
        runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME_YAML,
        encoding=_DEFAULT_ENCODING,
        serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
    ):
        serializer = _CfgFileInterface.serializer2enum(serializer)
        if serializer == _CfgSerializer.YAML:
            return _CfgYamlInterface.save_configs(
                data_cfg=data_cfg, model_cfg=model_cfg, runtime_cfg=runtime_cfg,
                dst_path=dst_path, encoding=encoding,
                data_config_filename=data_config_filename,
                model_config_filename=model_config_filename,
                runtime_config_filename=runtime_config_filename
            )
        elif serializer == _CfgSerializer.JSON:
            return _CfgJsonInterface.save_configs(
                data_cfg=data_cfg, model_cfg=model_cfg, runtime_cfg=runtime_cfg,
                dst_path=dst_path, encoding=encoding,
                data_config_filename=data_config_filename,
                model_config_filename=model_config_filename,
                runtime_config_filename=runtime_config_filename
            )
        else:
            raise NotImplementedError(f'Invalid serializer {serializer}')

    @staticmethod
    def from_files(
        src_path: str,
        data_config_filename: str = DEFAULT_D_CFG_FILENAME_YAML,
        model_config_filename: str = DEFAULT_MDL_CFG_FILENAME_YAML,
        runtime_config_filename: str = DEFAULT_RT_CFG_FILENAME_YAML,
        serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
        encoding=_DEFAULT_ENCODING
    ):
        d_cfg, m_cfg, r_cfg = ConfigurationManager.load_configs(
            src_path=src_path, encoding=encoding, serializer=serializer,
            data_config_filename=data_config_filename,
            model_config_filename=model_config_filename,
            runtime_config_filename=runtime_config_filename
        )
        return ConfigurationManager(d_cfg, m_cfg, r_cfg)

    def to_files(
            self,
            dst_dir_path: str,
            serializer: Union[str, _CfgSerializer] = _CfgSerializer.YAML,
            encoding: Optional[str] = None
    ) -> None:
        serializer = _CfgFileInterface.serializer2enum(serializer)
        d_cfg = self.data_config.inner
        mdl_cfg = self.model_config.inner
        rt_cfg = self.runtime_config.inner

        d_filename = self.data_config_filename
        mdl_filename = self.model_config_filename
        rt_filename = self.runtime_config_filename
        encoding = encoding or self.encoding

        if serializer == _CfgSerializer.YAML:
            return _CfgYamlInterface.save_configs(
                d_cfg, mdl_cfg, rt_cfg,
                dst_dir_path,
                d_filename, mdl_filename, rt_filename,
                encoding=encoding)
        elif serializer == _CfgSerializer.JSON:
            return _CfgJsonInterface.save_configs(
                d_cfg, mdl_cfg, rt_cfg,
                dst_dir_path,
                d_filename, mdl_filename, rt_filename,
                encoding=encoding)
        else:
            raise NotImplementedError(f'Invalid serializer {serializer}')

    def __init_role(self) -> None:
        self._role: Optional[Role] = None

    @property
    @Singleton.thread_safe_ensurance
    def role(self) -> Role:
        """return the role of this runtime entity.

        Raises:
            AttributeError: called without role configured.

        Returns:
            Role: the role of this runtime entity.
        """
        if self._role is None:
            raise AttributeError('the role of this node has not been set yet.')
        return self._role

    @role.setter
    @Singleton.thread_safe_ensurance
    def role(self, role: Role):
        """set the role of this runtime entity.
        This method should be called only once.
        It is recommoned to be set as soon as the role of this runtime could be known.

        Args:
            role (Role): the role which this entity should be.

        Raises:
            AttributeError: called more than once.
        """
        if self._role is not None:
            raise AttributeError('the role of a node can only be set once.')
        self._role = role
