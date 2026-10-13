"""Consistent vector circuits; explanations belong to adjacent article text."""
import os
from pathlib import Path
from paper_theme import export
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
N,O,T,G='#183b56','#bb8424','#257e83','#727a80'
BG={N:'#f2f6fa',T:'#eff8f7',O:'#fcf7ed',G:'#f5f6f7'}
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans','text.color':N})
def panel(a,s):
 a.set(xlim=(0,100),ylim=(0,8));a.axis('off');a.text(1,7.7,s,fontsize=11,fontweight='bold',va='center')
def wire(a,y,s,b=False):
 a.text(2,y,s,fontsize=11,va='center');a.plot([14,97],[y,y],color='#98a6b0',lw=.9,zorder=0)
 if b:a.plot([15.2,16.2],[y-.1,y+.1],color=G,lw=.9)
def gate(a,x,ys,s,c=N,w=12,dash=False,fs=11):
 lo,hi=min(ys),max(ys);pad=.15*8/(a.get_position().height*a.figure.get_size_inches()[1]);a.add_patch(FancyBboxPatch((x-w/2,lo-pad),w,hi-lo+2*pad,boxstyle='round,pad=0,rounding_size=.2',fc=BG[c],ec=c,lw=1.2,ls=(0,(3,2)) if dash else '-',zorder=3));a.text(x,(lo+hi)/2,s,fontsize=fs,color=N,ha='center',va='center',zorder=4)
def ctrl(a,x,y,ys,s,c=N,w=12,mux=False,dash=False):
 a.plot([x,x],[min(ys),y],color=c,lw=1.1,zorder=1);a.scatter([x],[y],s=22,marker='s' if mux else 'o',facecolors='white' if mux else c,edgecolors=c,linewidths=1.1,zorder=4);gate(a,x,ys,s,c,w,dash)
def make(h,labels):
 f,aa=plt.subplots(len(h),1,figsize=(8.4,sum(h)),gridspec_kw={'height_ratios':h});f.subplots_adjust(left=.02,right=.99,top=.98,bottom=.025,hspace=.22)
 for a,l in zip(aa,labels):panel(a,l)
 return f,aa
def save(f,s):
 for e in ['pdf','svg','png']:export(f,ROOT/f'outputs/figures/{s}.{e}',dpi=240,facecolor='white')
 plt.close(f)
f,aa=make([1.35,2],['(a)  $A_X$',r'(b)  $U_\ell(x)$']);a=aa[0]
wire(a,5.2,'$X$',True);wire(a,2.5,'$S$')
ctrl(a,23,5.2,[2.5],r'$R_z(\phi_x)$',G,14,True,True);ctrl(a,43,5.2,[2.5],'$U_0(x)$',N,14,True)
a.text(55,2.5,r'$\cdots$',fontsize=13,ha='center',va='center');ctrl(a,66,5.2,[2.5],'$U_{L-1}(x)$',N,16,True)
gate(a,83,[2.5],r'$B_{\mathbf{v}}$',T,10);gate(a,95,[2.5],'$X$',T,7)
a=aa[1]
for y,l in [(6.2,'$x_0$'),(4.2,'$x_1$'),(2.2,'$S$')]:wire(a,y,l)
gate(a,23,[2.2],r'$R_x(a_\ell)$',N,14);gate(a,42,[2.2],r'$R_y(\theta_\ell)$',N,15)
ctrl(a,61,6.2,[2.2],r'$R_y(b_\ell\Delta)$',N,17);ctrl(a,80,4.2,[2.2],r'$R_y(2b_\ell\Delta)$',N,18);gate(a,95,[2.2],r'$R_z(c_\ell)$',N,10,fs=10)
save(f,'C01_qru_loading')
f,aa=make([2.3,2.15],['(a)  Oracle','(b)  $E_X$']);a=aa[0];ys=[6.4,5.15,3.9,2.65,1.4]
for y,l,b in zip(ys,['$X$','$Y$','$S$','$a$','$T$'],[True,True,False,False,False]):wire(a,y,l,b)
gate(a,28,ys[:4],'$E_X$',T,16);ctrl(a,53,ys[1],[ys[-1]],'$V_j(g_Y)$',O,18,True);gate(a,77,ys[:4],r'$E_X^\dagger$',T,16);gate(a,94,ys[1:4],'trace',G,9,fs=9.5)
a=aa[1]
for y,l,b in [(6.2,'$y_0$',False),(4.7,'$y_1$',False),(3.2,'$X$',True),(1.7,'$S$',False)]:wire(a,y,l,b)
gate(a,25,[6.2,4.7],'$P_a$',T,12);gate(a,25,[3.2,1.7],'$A_X$',N,12)
for x,y in [(44,6.2),(62,4.7),(77,4.7)]:ctrl(a,x,y,[3.2,1.7],'$Q_X$',N,11)
gate(a,93,[6.2,4.7],r'$\mathsf{F}_4^\dagger$',T,10);save(f,'C02_service_estimator')
f,aa=make([1.95,1.95],['(a)  Controlled $Q_X$','(b)  $P_a$']);a=aa[0]
for y,l,b in [(6,'$p$',False),(4,'$X$',True),(2,'$S$',False)]:wire(a,y,l,b)
gate(a,24,[6],r'$P(\pi)$',N,12);ctrl(a,39,6,[2],'$Z$',N,8);ctrl(a,55,6,[4,2],r'$A_X^\dagger$',N,12);ctrl(a,72,6,[2],'$-Z$',N,10);ctrl(a,91,6,[4,2],'$A_X$',N,12)
a=aa[1]
for y,l in [(6,'$y_0$'),(4,'$y_1$'),(2,'$a$')]:wire(a,y,l)
gate(a,24,[6],'$X$',T,10);gate(a,24,[4],'$H$',T,10);gate(a,39,[4],'$Z$',T,10)
a.plot([52,52],[2,4],color=T,lw=1.1);a.scatter([52],[4],s=22,c=T,zorder=4);a.scatter([52],[2],s=72,facecolors='white',edgecolors=T,linewidths=1.2,zorder=4);a.scatter([52],[2],s=52,marker='+',c=T,linewidths=1.1,zorder=5)
gate(a,69,[6,4,2],r'$\mathsf{F}_8$',T,14);gate(a,85,[2],'$H$',T,10);gate(a,96,[2],'$X$',T,7);save(f,'C03_grover_window')
f,aa=plt.subplots(1,2,figsize=(8.4,2.6));f.subplots_adjust(left=.02,right=.99,top=.97,bottom=.03,wspace=.2)
for a,l in zip(aa,['(a)  Direct','(b)  Signed']):panel(a,l)
a=aa[0];wire(a,6,'$Y$',True);wire(a,2,'$T$');ctrl(a,55,6,[2],r'$R_z(2\gamma_j g_y)$',O,60,True)
a=aa[1]
for y,l,b in [(6,'$Y$',True),(4,'$R$',True),(2,'$T$',False)]:wire(a,y,l,b)
gate(a,27,[6,4],'$D_b$',O,19);ctrl(a,56,4,[2],r'$R_z(2\gamma_j g_{y,b})$',O,62,True);gate(a,87,[6,4],r'$D_b^\dagger$',O,19);save(f,'C04_target_control')
