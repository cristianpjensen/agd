from .mlp_gate import MLPGateCFGAdapter
from .learned_pos_emb import LearnedPosEmbedCFGAdapter
from .additive import AdditiveCFGAdapter
from .attention import CrossAttentionCFGAdapter


ADAPTER_MODELS = {
    "mlp_gate": MLPGateCFGAdapter,
    "learned_pos_emb": LearnedPosEmbedCFGAdapter,
    "additive": AdditiveCFGAdapter,
    "attention": CrossAttentionCFGAdapter,
}
