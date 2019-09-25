
# Multiagent ASN

## Installation instructions

This project based on [SMAC](https://github.com/oxwhirl/smac) and [pymarl](https://github.com/oxwhirl/pymarl)

OS: Ubuntu 14 or 16 

Install StarCraftII:

```shell
chmod 755 ./install_sc2.sh
./install_sc2.sh
tar -xvf SMAC_Maps_supply.tar.gz
mv SMAC_Maps/* 3rdparty/StarCraftII/Maps/SMAC_Maps/
```

You must install anaconda before installing code env.

Install Code Env: 

```shell
conda create -n ASN python=3.6
source activate ASN
conda install certifi==2018.8.24
pip install -r requirements.txt
```

## Run an experiment 

Activate Code Env:

```shell
source activate ASN
export PYTHONPATH=. 
```



Then:

### 8m:

#### IQL

##### Vanilla
```shell
python src/main.py --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='dense_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```

##### Attention
```shell
python src/main.py --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='dense_rnn_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```


##### Entity-Attention
```shell
python src/main.py --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='dense_rnn_entity_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```

##### Dueling
```shell
python src/main.py --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='dense_rnn_dueling' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```

##### ASN
```shell
python src/main.py --config=iql_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='asn_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```

#### VDN

##### Vanilla
```shell
python src/main.py --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='dense_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```

##### Attention
```shell
python src/main.py --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='dense_rnn_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```

##### Entity-Attention
```shell
python src/main.py --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='dense_rnn_entity_attention' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```

##### Dueling
```shell
python src/main.py --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='dense_rnn_dueling' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```

##### ASN
```shell
python src/main.py --config=vdn_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 agent='asn_rnn' legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100
```

#### QMIX

##### Vanilla
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 env='sc2' seed=100 agent='dense_rnn'
```

##### Attention
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 env='sc2' seed=100 agent='dense_rnn_attention'
```

##### Entity-Attention
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 env='sc2' seed=100 agent='dense_rnn_entity_attention'
```

##### Dueling
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 env='sc2' seed=100 agent='dense_rnn_dueling'
```

##### ASN
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 env='sc2' seed=100 agent='asn_rnn'
```


### -1-Paddings

#### QMIX

##### Vanilla
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='dense_rnn' env='sc2_not_0'
```

##### Attention
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='dense_rnn_attention' env='sc2_not_0'
```

##### Entity-Attention
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='dense_rnn_entity_attention' env='sc2_not_0'
```

##### Dueling
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='dense_rnn_dueling' env='sc2_not_0'
```

##### ASN
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='asn_rnn' env='sc2_not_0'
```


### 1-Paddings

#### QMIX

##### Vanilla
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='dense_rnn' env='sc2_set_1'
```

##### Attention
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='dense_rnn_attention' env='sc2_set_1'
```

##### Entity-Attention
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='dense_rnn_entity_attention' env='sc2_set_1'
```

##### Dueling
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='dense_rnn_dueling' env='sc2_set_1'
```

##### ASN
```shell
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=8m_8m agents_num=8 enemies_num=8 legal_action=False batch_size_run=1 use_tensorboard=True save_model=True runner_log_interval=2000 seed=100 agent='asn_rnn' env='sc2_set_1'
```

