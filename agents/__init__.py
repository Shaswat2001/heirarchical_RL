from .gcbc import GCBCAgent, GCBC_CONFIG_DICT
from .gciql import GCIQLAgent, GCIQL_CONFIG_DICT
from .hiql import HIQLAgent, HIQL_CONFIG_DICT
from .hbc import HBCAgent, HBC_CONFIG_DICT
agents = {
    "gcbc": (GCBCAgent, GCBC_CONFIG_DICT),
    "gciql": (GCIQLAgent, GCIQL_CONFIG_DICT),
    "hiql": (HIQLAgent, HIQL_CONFIG_DICT),
    "hbc": (HBCAgent, HBC_CONFIG_DICT)
}