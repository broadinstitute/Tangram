# install Spapros via https://github.com/theislab/spapros

import spapros as sp

def spapros(adata_sc):
    selector = sp.se.ProbesetSelector(adata_sc)
    selector.select_probeset()
    return selector.probeset.index[selector.probeset["selection"]].to_list()
