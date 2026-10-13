"""Minimal vector overview; all graphical elements are schematic."""
import os
from pathlib import Path
from paper_theme import export
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Arc, Wedge
ROOT=Path(__file__).resolve().parents[1]
NAVY,OCHRE,TEAL,GRAY='#183b56','#bb8424','#257e83','#727a80'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','text.color':NAVY,'mathtext.fontset':'dejavusans'})
fig,ax=plt.subplots(figsize=(9,4.8));fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.97)
ax.set(xlim=(0,100),ylim=(0,51));ax.axis('off')
def text(x,y,s,size=12,color=NAVY,weight='normal'):
    ax.text(x,y,s,fontsize=size,color=NAVY,ha='center',va='center',fontweight=weight)
def arrow(a,b,col=GRAY,lw=1.6,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=col,lw=lw,connectionstyle=f'arc3,rad={rad}'))
def bloch(x,y,r,col=TEAL):
    ax.add_patch(Circle((x,y),r,fill=False,ec=col,lw=1.5))
    ax.add_patch(Ellipse((x,y),2*r,.65*r,fill=False,ec=col,lw=1))
    ax.plot([x,x],[y-r,y+r],color=col,lw=.8,alpha=.55)
# Four actions, one clear left-to-right path.
xs=[12,37,62,87]
for x in xs:ax.add_patch(Circle((x,37),9,fc='#f3f7fa',ec='none'))
for x in [22,47,72]:arrow((x,37),(x+5,37),NAVY)
bloch(12,37,5,TEAL);arrow((12,37),(15.3,41),NAVY,2)
text(12,48,r'$\sum_x c_x|x\rangle$',12)
for i,h in enumerate([1.2,2.4,5.5,8,4.4,1.8]):
    ax.add_patch(FancyBboxPatch((31+i*2,33),1.4,h,boxstyle='round,pad=0,rounding_size=.35',fc=TEAL,ec='none'))
bloch(62,37,5,OCHRE);arrow((62,37),(65.2,40.6),OCHRE,2)
ax.add_patch(Arc((62,37),13,13,theta1=30,theta2=150,color=OCHRE,lw=1.8))
arrow((56.4,40.1),(56.1,39.3),OCHRE,1.5)
# Return arrow and residual workspace dots, without claiming perfect reset.
arrow((89,41),(82,34),TEAL,2,rad=.5)
for x,y,col in [(87,34,TEAL),(90,34,TEAL),(87,31,TEAL),(90,31,OCHRE)]:
    ax.add_patch(Circle((x,y),.8,fc=col,ec='none'))
for x,s in zip(xs,['QRU','Estimate','Control','Invert']):text(x,24.2,s,12,weight='bold')
ax.plot([6,94],[20,20],color='#dce5eb',lw=1)
# Three research questions: channel distance, coherent memory, lift sensitivity.
for x in [19,50,81]:ax.add_patch(Circle((x,10.5),6.8,fc='#f8fafb',ec='none'))
ax.add_patch(Circle((17.8,10.5),3.8,fill=False,ec=NAVY,lw=1.5))
ax.add_patch(Circle((20.2,10.5),3.8,fill=False,ec=OCHRE,lw=1.5,ls=(0,(3,2))))
arrow((17.8,10.5),(20.2,10.5),NAVY,1)
arrow((46,10.5),(54,10.5),OCHRE,1.8,rad=.75)
arrow((54,10.5),(46,10.5),NAVY,1.8,rad=.75)
text(50,10.5,r'$W$',13)
ax.add_patch(Wedge((81,10.5),4.5,270,360,fc=OCHRE,alpha=.25,ec='none'))
bloch(81,10.5,4.5,TEAL)
arrow((81,10.5),(83.5,13.6),TEAL,1.7)
ax.add_patch(Ellipse((81,10.5),3,9,angle=30,fill=False,ec=OCHRE,lw=1.2))
for x,s,col in [(19,'Channel error',NAVY),(50,'Workspace memory',OCHRE),(81,'Fibre boundary',TEAL)]:text(x,1.5,s,11,col,weight='bold')
for ext in ['pdf','svg','png']:export(fig,ROOT/f'outputs/figures/00_research_overview.{ext}',dpi=240,facecolor='white')
plt.close(fig)
