"""
Draw a single grid attention figure.

python draw_one_grid.py \
    --ckpt /share/dean/mx253/icl_ca/checkpoint/ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-d512-lr1e-3-wd0.2-ep500/ca-cmixed-L16_T10_M4_seed42_30k-layers2-heads-1-1-d512-lr1e-3-wd0.2-ep500_best.pt \
    --test_path /share/dean/mx253/icl_ca/eca_data/L16_T10_M4_seed42_30k/mixed_test.npz \
    --cell_width 15 \
    --num_context_rows 4 \
    --sample_idx 16796
"""
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_generate import ECADataset
from utils import extract_attention_maps
from error_analysis import load_model

COLOR_QUERY='#3b82f6'; COLOR_NB='#e24b4a'
COLOR_MATCH_OUT='#16a34a'; COLOR_MATCH_PAR='#86efac'

def draw(ax, cv, h_attn, qfi, cw, stride, title, nb_set, mo_set, mp_set):
    T=len(cv); nrows=(T+stride-1)//stride
    qr=qfi//stride; qc=qfi%stride; ac=(qc+1)%cw
    vis=[r*stride+c for r in range(nrows) for c in range(cw)
         if r*stride+c<qfi and r*stride+c<T]
    max_a=max([h_attn[i] for i in vis]) if vis else 1e-8

    for r in range(nrows):
        for c in range(cw):
            flat=r*stride+c
            if flat>=T: continue
            x=c; y=nrows-1-r
            is_q=(r==qr and c==ac); is_f=flat>qfi
            a=h_attn[flat] if flat<len(h_attn) else 0
            t=min(a/max_a,1.0) if max_a>0 else 0

            if is_q:
                bg='#dbeafe'; tc='#1e40af'; dv='?'
            elif is_f:
                bg='#f5f5f5'; tc='#ccc'; dv=str(int(cv[flat]))
            else:
                rc=int(247-t*215)/255; gc=int(249-t*220)/255; bc=int(253-t*120)/255
                bg=(rc,gc,bc); tc='#fff' if t>0.5 else '#333'; dv=str(int(cv[flat]))

            ax.add_patch(mpatches.FancyBboxPatch(
                (x+0.04,y+0.04),0.92,0.92,boxstyle="round,pad=0.01",
                facecolor=bg,edgecolor='#e0e0e0',linewidth=0.15))

            if is_q:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x+0.02,y+0.02),0.96,0.96,boxstyle="round,pad=0.01",
                    facecolor='none',edgecolor=COLOR_QUERY,linewidth=1.5))
            elif (r,c) in nb_set:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x+0.02,y+0.02),0.96,0.96,boxstyle="round,pad=0.01",
                    facecolor='none',edgecolor=COLOR_NB,linewidth=1.2))
            elif (r,c) in mo_set:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x+0.02,y+0.02),0.96,0.96,boxstyle="round,pad=0.01",
                    facecolor='none',edgecolor=COLOR_MATCH_OUT,linewidth=1.2))
            elif (r,c) in mp_set:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x+0.02,y+0.02),0.96,0.96,boxstyle="round,pad=0.01",
                    facecolor='none',edgecolor=COLOR_MATCH_PAR,linewidth=1.0))

            ax.text(x+0.5, y+0.5, dv, ha='center', va='center',
                    fontsize=5, fontweight='semibold', color=tc)

        ax.text(-0.3, nrows-1-r+0.5, f't{r}', ha='right', va='center',
                fontsize=5, color='#999')

    ax.set_xlim(-0.5, cw+0.2)
    ax.set_ylim(-0.2, nrows+0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9, pad=6)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--ckpt",required=True)
    pa.add_argument("--test_path",required=True)
    pa.add_argument("--cell_width",type=int,required=True)
    pa.add_argument("--num_context_rows",type=int,default=4)
    pa.add_argument("--sample_idx",type=int,required=True)
    args=pa.parse_args()

    cw=args.cell_width; stride=cw+1
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,cfg=load_model(args.ckpt,dev)
    ds=ECADataset(args.test_path)

    si=args.sample_idx
    X,y,mask=ds[si]
    last_cell=y[-1:].numpy()
    cv=np.concatenate([X.numpy(),last_cell])

    # Find query position
    eps=[p for p in range(len(mask)) if mask[p].item()>0 and p%stride<cw]
    qpos=eps[len(eps)//2]
    qr=qpos//stride; qc=qpos%stride; ac=(qc+1)%cw

    # Extract attention
    attn_maps,preds=extract_attention_maps(model,X.unsqueeze(0),dev)

    # Build annotation sets
    nr=qr-1; nbc=[(ac-1)%cw,ac,(ac+1)%cw]
    nb_set=set(); qpat=None; mo_set=set(); mp_set=set()
    if nr>=0:
        for c in nbc: nb_set.add((nr,c))
        pb=[int(cv[nr*stride+c]) for c in nbc]; qpat=tuple(pb)
        for r in range(0,nr):
            for c in range(cw):
                l=(c-1)%cw; ri=(c+1)%cw
                if r*stride+ri>=len(cv): continue
                bits=(int(cv[r*stride+l]),int(cv[r*stride+c]),int(cv[r*stride+ri]))
                if bits==qpat:
                    mp_set.update([(r,l),(r,c),(r,ri)])
                    mo_set.add((r+1,c))
        mo_set-=nb_set; mp_set-=nb_set; mp_set-=mo_set

    # Draw
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.4))
    for li in range(2):
        ha=attn_maps[li].mean(axis=0)[qpos,:]
        draw(axes[li],cv,ha,qpos,cw,stride,
             'Layer 1' if li==0 else 'Layer 2',
             nb_set,mo_set,mp_set)

    lh=[mpatches.Patch(facecolor='#dbeafe',edgecolor=COLOR_QUERY,linewidth=1.5,label='Query cell'),
        mpatches.Patch(facecolor='none',edgecolor=COLOR_NB,linewidth=1.2,label='Neighborhood'),
        mpatches.Patch(facecolor='none',edgecolor=COLOR_MATCH_OUT,linewidth=1.2,label='Matched output'),
        mpatches.Patch(facecolor='none',edgecolor=COLOR_MATCH_PAR,linewidth=1.0,label='Matched neighborhood')]
    fig.legend(handles=lh,loc='lower center',ncol=4,fontsize=6,frameon=False,
               bbox_to_anchor=(0.5,-0.01))
    plt.tight_layout()

    rule=int(ds.rules[si]) if ds.rules is not None else -1
    tv=int(y[qpos].item()); pv=int(preds[qpos])
    ps="".join(str(b) for b in pb) if qpat else "x"
    fname=f'fig_grid_s{si}_rule{rule}_t{qr}c{ac}_pat{ps}_pred{pv}_true{tv}'
    plt.savefig(fname+'.pdf',bbox_inches='tight',pad_inches=0.02,dpi=600)
    plt.savefig(fname+'.png',bbox_inches='tight',pad_inches=0.02,dpi=600)
    plt.close()
    print(f"Saved {fname}.pdf/png")

if __name__=="__main__": main()