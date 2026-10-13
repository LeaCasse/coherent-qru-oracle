"""Shared publication palette and non-colour encodings.
Every categorical plot must use markers/line patterns in addition to colour.
Schematics use explicit gate/register labels and topology.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
NAVY,OCHRE,TEAL,SLATE,CHARCOAL='#183b56','#bb8424','#257e83','#727a80','#454b50'
COLORS=[NAVY,OCHRE,TEAL,SLATE,CHARCOAL]
MARKERS=['o','s','^','D','v']
STYLES=['-','--','-.',':',(0,(5,2,1,2))]
COLOR_STYLE=dict(zip(COLORS,zip(MARKERS,STYLES)))
def setup():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.prop_cycle':plt.cycler(color=COLORS),'legend.frameon':False})
def finish(fig,path):
 """Save exact data as vector graphics and a raster preview."""
 fig.tight_layout()
 for ext in ['pdf','svg','png']:
  export(fig,Path(path).with_suffix('.'+ext),bbox_inches='tight',dpi=240,facecolor='white')
 plt.close(fig)

def export(fig,path,**kwargs):
 """Atomically publish a fully closed figure file."""
 import os
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name('_tmp_'+path.name)
 fig.savefig(temporary,**kwargs)
 os.replace(temporary,path)
