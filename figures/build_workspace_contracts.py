"""Redraw workspace contracts without changing their operational content."""
import os
from pathlib import Path
from paper_theme import export
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
NAVY,OCHRE,TEAL,GRAY='#183b56','#bb8424','#257e83','#727a80'

def build():
 plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
 fig,ax=plt.subplots(figsize=(10,4.7));fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.98)
 ax.set(xlim=(0,100),ylim=(0,49));ax.axis('off')
 def txt(x,y,s,size=11,col=NAVY,weight='normal',ha='center'):
  ax.text(x,y,s,fontsize=size,color=NAVY,fontweight=weight,ha=ha,va='center')
 def box(x,y,s,col=NAVY,dashed=False,w=10):
  ax.add_patch(FancyBboxPatch((x-w/2,y-2.9),w,5.8,boxstyle='round,pad=0,rounding_size=.85',fc='white',ec=col,lw=1.25,linestyle='--' if dashed else '-'))
  txt(x,y,s,10 if '\n' in s else 13,col)
 def arrow(a,b,col):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=1.1,color=col))
 rows=[(39,OCHRE,'Shared work','#fcf8ef'),(24,NAVY,'Fresh work','#f3f6fa'),(9,TEAL,'Input preserved','#f0f7f6')]
 for y,col,label,bg in rows:
  ax.add_patch(FancyBboxPatch((1,y-5.5),98,11,boxstyle='round,pad=0,rounding_size=1.4',fc=bg,ec='none'))
  ax.plot([3,3],[y-3,y+3],color=col,lw=2.5,solid_capstyle='round')
  txt(5,y,label,11,col,'bold','left')
 xs=[29,43,57,71,85]
 for y,col in [(39,OCHRE),(24,NAVY)]:
  for a,b in zip(xs[:-1],xs[1:]):arrow((a+5.4,y),(b-5.4,y),col)
 for x,s in zip(xs,[r'$E_X$',r'$V_1(g_Y)$',r'$R$',r'$V_2(g_Y)$',r'$E_X^\dagger$']):box(x,39,s,OCHRE)
 arrow((90.5,39),(93,39),OCHRE);txt(95.5,39,'trace',9.5)
 for x,s in zip(xs,[r'$W_1$','trace\n+ renew',r'$R$',r'$W_2$','trace']):box(x,24,s,NAVY,s.startswith('trace'))
 box(29,9,r'$E_X$',TEAL);box(57,9,r'$F_X(g_Y)$',TEAL,w=18);box(85,9,r'$E_X^\dagger$',TEAL)
 arrow((34.5,9),(47.5,9),TEAL);arrow((66.5,9),(79.5,9),TEAL)
 arrow((90.5,9),(93,9),TEAL);txt(95.5,9,'trace',9.5,TEAL)
 for ext in ['pdf','svg','png']:export(fig,ROOT/f'outputs/figures/01_contract.{ext}',dpi=240,facecolor='white')
 plt.close(fig)
if __name__=='__main__':build()
