
from uniperceiver.utils.registry import Registry

DECODE_STRATEGY_REGISTRY = Registry("DECODE_STRATEGY")
DECODE_STRATEGY_REGISTRY.__doc__ = """
Registry for decode strategy
"""

def build_beam_searcher(cfg):
    beam_search = None if cfg.DECODE_STRATEGY.NAME.lower() == "none" else DECODE_STRATEGY_REGISTRY.get(cfg.DECODE_STRATEGY.NAME)(cfg)
    return beam_search

def build_greedy_decoder(cfg):
    greedy_decoder = DECODE_STRATEGY_REGISTRY.get("GreedyDecoder")(cfg)
    return greedy_decoder