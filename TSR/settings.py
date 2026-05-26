import os
import torch
settings_path = os.path.dirname(__file__)

DATA_ROOT = os.path.join(settings_path, 'data')
CACHE_ROOT = os.path.join(settings_path, 'cache')
RESULT_ROOT = os.path.join(settings_path, 'result')


GRAPH_ICEWS18 = "ICEWS18"
GRAPH_ICEWS14 = "ICEWS14"
GRAPH_ICEWS05_15 = "ICEWS05_15"
GRAPH_GDELT = "GDELT26"
ALL_GRAPHS = [GRAPH_ICEWS18, GRAPH_ICEWS14, GRAPH_ICEWS05_15, GRAPH_GDELT]

DGL_GRAPH_ID_TYPE = torch.int32
INTER_EVENT_TIME_DTYPE = torch.float32
