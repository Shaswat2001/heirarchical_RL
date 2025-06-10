from .gcbc import GCBCAgent, GCBC_CONFIG_DICT
from .gciql import GCIQLAgent, GCIQL_CONFIG_DICT

agents = {
    "gcbc": (GCBCAgent, GCBC_CONFIG_DICT),
    "gciql": (GCIQLAgent, GCIQL_CONFIG_DICT)
}