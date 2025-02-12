# install SpatialDE2 via https://github.com/PMBio/SpatialDE
import sys
sys.path.insert(1, '../../SpatialDE')

import SpatialDE

def svg(adata_st):
    adata_st.X = adata_st.raw.X
    svg_full, _ = SpatialDE.test(adata_st, omnibus=True)
    return svg_full[svg_full.padj < 0.05].gene
