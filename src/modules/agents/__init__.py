REGISTRY = {}

from .rnn_agent import RNNAgent

from .asn_agent import AsnAgent
from .asn_rnn_agent import AsnRNNAgent
from .asn_diff_type_agent import AsnDiffAgent
from .asn_diff_type_rnn_agent import AsnDiffRnnAgent
from .asn_wo_share_diff_type_agent import AsnDiffWoShareAgent

from .dense_agent import DenseAgent
from .dense_rnn_agent import DenseRNNAgent
from .dense_rnn_dueling_agent import DenseRNNDuelingAgent
from .dense_rnn_attention_agent import DenseRNNAttentionAgent
from .dense_rnn_entity_attention_agent import DenseRNNEntityAttentionAgent


REGISTRY["rnn"] = RNNAgent

# modify add agent type
REGISTRY["asn"] = AsnAgent
REGISTRY['asn_rnn'] = AsnRNNAgent
REGISTRY['asn_diff'] = AsnDiffAgent
REGISTRY['asn_wo_share_diff'] = AsnDiffWoShareAgent
REGISTRY['asn_diff_rnn'] = AsnDiffRnnAgent

REGISTRY['dense'] = DenseAgent
REGISTRY['dense_rnn'] = DenseRNNAgent
REGISTRY['dense_rnn_dueling'] = DenseRNNDuelingAgent
REGISTRY['dense_rnn_attention'] = DenseRNNAttentionAgent
REGISTRY['dense_rnn_entity_attention'] = DenseRNNEntityAttentionAgent