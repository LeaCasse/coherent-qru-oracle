"""Redraw paper figures from the archived CSVs; no new experimental runs."""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from paper_theme import setup, finish, CHARCOAL
setup()
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs/figures"
NAVY, OCHRE, TEAL, GRAY = "#183b56", "#bb8424", "#257e83", "#727a80"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "axes.labelcolor": NAVY, "text.color": NAVY,
})


def read(folder, name):
    with (Path(os.environ.get("QRU_DATA_DIR", ROOT / "reference_data")) / folder / name).open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    finish(fig, FIG / name)


def run():
    FIG.mkdir(exist_ok=True)
    from build_workspace_contracts import build as build_contract
    build_contract()
    direct = read("direct_control","channels.csv")+read("direct_control","calibrated.csv")
    fig, axes=plt.subplots(1,2,figsize=(8.4,3.2))
    for ax,depth in zip(axes,[2,4]):
        for policy,label,color,style in [
            ("signed4",r"signed $b=4$",GRAY,":"),
            ("signed8",r"signed $b=8$",TEAL,"-."),
            ("direct","direct",NAVY,"-"),
            ("calibrated","calibrated",OCHRE,"--"),
        ]:
            means=[];low=[];high=[]
            for m in [2,3,4,5]:
                values=np.array([float(r["diamond_upper"]) for r in direct
                                 if int(r["depth"])==depth and int(r["phase_bits"])==m
                                 and r["policy"]==policy and r["window"]=="sine"])
                draws=np.random.default_rng(1701).choice(values,(5000,len(values)),replace=True).mean(1)
                means.append(values.mean());q=np.quantile(draws,[.025,.975])
                low.append(q[0]);high.append(q[1])
            ax.plot([2,3,4,5],means,style,marker={GRAY:"D",TEAL:"^",NAVY:"o",OCHRE:"s"}[color],markersize=4,label=label,color=color)
            ax.fill_between([2,3,4,5],low,high,color=color,alpha=.09)
        ax.set(yscale="log",xlabel=r"Phase width $m$",ylabel="Full diamond error",
               title=f"({chr(97+depth//2-1)}) depth {depth}")
        ax.set_xticks([2,3,4,5]);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    save(fig,"02_precision")
    process=read("process_study","channels.csv")
    protocols=["repeated_z","echo","alternating","input_mixed_echo"]
    titles=["(a) repeated Z","(b) target echo","(c) alternating Z, X","(d) input-mixed echo"]
    fig,axes=plt.subplots(2,2,figsize=(8.4,5.2),sharex=True,sharey=True)
    for name,title,ax in zip(protocols,titles,axes.ravel()):
        for mode,color in [("shared",OCHRE),("fresh",NAVY)]:
            means=[];sd=[]
            for k in [2,4,8]:
                values=np.array([float(r["choi_trace_norm"]) for r in process
                                 if r["depth"]=="2" and r["phase_bits"]=="4"
                                 and r["protocol"]==name and r["mode"]==mode
                                 and int(r["repetitions"])==k])
                means.append(values.mean());sd.append(values.std(ddof=1))
            ax.errorbar([2,4,8],means,yerr=sd,marker="s" if mode=="shared" else "o",linestyle="--" if mode=="shared" else "-",color=color,capsize=3,label=mode)
        ax.set_title(title);ax.set_xticks([2,4,8]);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    for ax in axes[1]:ax.set_xlabel(r"Uses $k$")
    for ax in axes[:,0]:ax.set_ylabel("Normalized Choi error")
    save(fig,"03_memory")
    gauge=read("process_study","gauge.csv")
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.2))
    for applicable,marker,color,label in [
        ("True","o",NAVY,"input preserved"),("False","x",OCHRE,"input mixed"),
    ]:
        selected=[r for r in gauge if r["input_preserved"]==applicable]
        axes[0].scatter([float(r["sector_bound"]) for r in selected],
                        [float(r["diamond_upper"]) for r in selected],
                        marker=marker,color=color,label=label)
    axes[0].plot([0,1.7],[0,1.7],"--",color=GRAY,lw=1)
    axes[0].set(xlabel=r"Sector bound $2p_1$",ylabel="Diamond fibre sensitivity",
                xlim=(-.04,1.7),ylim=(-.02,.7),title="(a) contract boundary")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    chosen=sorted([r for r in gauge if r["seed"]=="603" and r["protocol"]=="input_mixed_echo"],
                  key=lambda r:float(r["phase"]))
    axes[1].plot([0]+[float(r["phase"]) for r in chosen],
                 [0]+[float(r["diamond_lower"]) for r in chosen],
                 "o-",color=OCHRE,label="physical lower witness")
    axes[1].axhline(float(chosen[0]["sector_bound"]),color=NAVY,ls="--",
                   label=r"$2p_1$ (not applicable)")
    axes[1].set(xlabel=r"Branch fibre phase $\phi$",ylabel="Sensitivity",
                title="(b) seed 603, input mixed")
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.15)
    save(fig,"04_fibre")
    noise=read("process_study","gate_noise.csv")
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.1))
    for kind,color in [("unfused",OCHRE),("fused",NAVY)]:
        means=[]
        for p in [1e-5,1e-4,1e-3]:
            values=[float(r["noise_increment_trace_norm"]) for r in noise
                    if r["implementation"]==kind and float(r["cx_depolarizing_parameter"])==p]
            means.append(np.mean(values))
        axes[0].plot([1e-5,1e-4,1e-3],means,"s--" if kind=="unfused" else "o-",label=kind,color=color)
        count=int(next(r["cx_count"] for r in noise if r["implementation"]==kind))
        axes[1].bar(kind,count,color=color,edgecolor=CHARCOAL,linewidth=.8,hatch="///" if kind=="unfused" else "..");axes[1].text(kind,count+25,str(count),ha="center")
    axes[0].set(xscale="log",xlabel="CX depolarizing parameter",ylabel="Choi perturbation")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    axes[1].set(ylabel="Compiled CX",ylim=(0,1600))
    save(fig,"05_noise")
    fig,ax=plt.subplots(figsize=(6.5,3.0))
    for depth,color in [("2",NAVY),("4",OCHRE)]:
        means=[]
        for m in [2,3,4]:
            values=[float(r["choi_trace_norm"]) for r in process if r["depth"]==depth
                    and int(r["phase_bits"])==m and r["protocol"]=="alternating"
                    and r["mode"]=="shared" and r["repetitions"]=="8"]
            means.append(np.mean(values))
        ax.plot([2,3,4],means,"o-" if depth=="2" else "s--",color=color,label=f"depth {depth}")
    ax.set(xlabel=r"Phase width $m$",ylabel="Normalized Choi error")
    ax.set_xticks([2,3,4]);ax.legend();ax.grid(alpha=.15)
    save(fig,"A01_depth")
    fig,ax=plt.subplots(figsize=(6.5,3.0))
    for name,color in zip(protocols[:3],[NAVY,TEAL,OCHRE]):
        selected=[r for r in process if r["protocol"]==name and r["mode"]=="shared"
                  and r["has_diamond_sdp"]=="True" and name!="echo"]
        if not selected:continue
        ax.scatter([float(r["smooth_protocol_bound"]) for r in selected],
                   [float(r["diamond_upper"]) for r in selected],
                   color=color,marker={NAVY:"o",TEAL:"^",OCHRE:"s"}[color],label=name,alpha=.7)
    ax.plot([.005,2],[.005,2],"--",color=GRAY,lw=1)
    ax.set(xscale="log",yscale="log",xlabel="Smooth-protocol bound",ylabel="Diamond upper witness")
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    save(fig,"A02_bounds")
    resources=read("direct_control","resources.csv")
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.0))
    for policy,label,color in [
        ("signed4",r"signed $b=4$",GRAY),
        ("signed8",r"signed $b=8$",TEAL),
        ("multiplex4","matched multiplexor",OCHRE),
        ("direct","direct",NAVY),
    ]:
        selected=sorted([r for r in resources if r["policy"]==policy
                         and r["window"]=="sine"],key=lambda r:int(r["phase_bits"]))
        for ax,key in zip(axes,["qubits","cx"]):
            ax.plot([int(r["phase_bits"]) for r in selected],
                    [int(r[key]) for r in selected],{GRAY:"D:",TEAL:"^-.",OCHRE:"s--",NAVY:"o-"}[color],color=color,label=label,markersize=6 if policy=="multiplex4" else 4,markerfacecolor="white" if policy=="multiplex4" else color,linewidth=2.2 if policy=="multiplex4" else 1.4)
            ax.set_xlabel(r"Phase width $m$");ax.set_xticks([2,3,4,5])
            ax.grid(alpha=.15)
    axes[0].set_ylabel("Service qubits");axes[1].set_ylabel("Compiled CX")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.065),ncol=4,fontsize=8)
    save(fig,"A_resources")
    cal=read("direct_control","calibrated.csv")
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.0))
    for depth,color in [("2",NAVY),("4",OCHRE)]:
        selected=[r for r in cal if r["depth"]==depth]
        axes[0].scatter([float(r["bias_variance_bound"]) for r in selected],
                        [float(r["diamond_upper"]) for r in selected],
                        color=color,marker="o" if depth=="2" else "s",s=20,label=f"depth {depth}")
        axes[1].scatter([float(r["sector_bound"]) for r in selected],
                        [float(r["gauge_upper"]) for r in selected],color=color,marker="o" if depth=="2" else "s",s=20)
    for ax in axes:
        ax.plot([.001,2],[.001,2],"--",color=GRAY,lw=1)
        ax.set_xscale("log");ax.set_yscale("log");ax.grid(alpha=.15)
    axes[0].set(xlabel=r"Bound $2\gamma^2\mu$",ylabel="Diamond control error")
    axes[1].set(xlabel=r"Sector bound $2p_1$",ylabel="Diamond fibre sensitivity")
    axes[0].legend(fontsize=8)
    save(fig,"A_calibration")
    reuse=read("direct_control","reuse.csv")
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.0))
    for ax,m in zip(axes,[3,5]):
        for mode,color in [("fresh",NAVY),("same_work",OCHRE)]:
            means=[]
            for k in [1,2,4,8,16,32]:
                values=[float(r["diamond_upper"]) for r in reuse
                        if r["phase_bits"]==str(m) and r["mode"]==mode
                        and r["repetitions"]==str(k)]
                means.append(np.mean(values))
            ax.plot([1,2,4,8,16,32],means,"s--" if mode=="same_work" else "o-",color=color,
                    label="shared" if mode=="same_work" else "fresh")
        ax.set(xscale="log",yscale="log",xlabel=r"Uses $k$",
               ylabel="Full diamond error",title=rf"$m={m}$")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    save(fig,"A_reuse")
    print("Redrew nine numerical figures from unchanged source data.")


if __name__=="__main__":
    run()
