"""
Generate paper figures from a trained ECA transformer checkpoint.

Usage:
    python figures.py \
        --ckpt /path/to/best.pt \
        --test_path /path/to/test.npz \
        --cell_width 15 \
        --num_context_rows 4 \
        --output_dir ./figures
"""
import argparse, os, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from data_generate import ECADataset
from models import VSimpleTransformer
from utils import extract_attention_maps
from error_analysis import load_model, extract_layer_outputs, get_parent_pattern

plt.rcParams.update({
    'font.family':'sans-serif','font.size':8,'axes.titlesize':9,
    'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'figure.dpi':600,'savefig.dpi':600,
    'savefig.bbox':'tight','savefig.pad_inches':0.05,
    'axes.linewidth':0.4,'xtick.major.width':0.4,'ytick.major.width':0.4,
})
TAB10=['#4E79A7','#F28E2B','#E15759','#76B7B2','#59A14F','#EDC948','#B07AA1','#FF9DA7']
CMAP_L1='Blues'; CMAP_L2='Greens'; CMAP_OPT='Oranges'
COLOR_QUERY='#3b82f6'; COLOR_NB='#e24b4a'; COLOR_MATCH_OUT='#16a34a'; COLOR_MATCH_PAR='#86efac'

# ============ Helpers ============
def get_cell_categories_detailed(cv, q_pos, cw, stride):
    qr=q_pos//stride; qc=q_pos%stride; ac=(qc+1)%cw
    pL=set();pC=set();pR=set();po=set();pp=set()
    nr=qr-1
    if nr<0: return pL,pC,pR,po,pp
    lc=(ac-1)%cw; cc=ac; rc=(ac+1)%cw
    pL.add(nr*stride+lc);pC.add(nr*stride+cc);pR.add(nr*stride+rc)
    ap=pL|pC|pR
    qpat=(int(cv[nr*stride+lc]),int(cv[nr*stride+cc]),int(cv[nr*stride+rc]))
    for r in range(0,nr):
        for c in range(cw):
            l=(c-1)%cw;ri=(c+1)%cw
            li=r*stride+l;ci=r*stride+c;rii=r*stride+ri
            if rii>=len(cv):continue
            bits=(int(cv[li]),int(cv[ci]),int(cv[rii]))
            if bits==qpat:
                pp.update([li,ci,rii])
                oi=(r+1)*stride+c
                if oi<len(cv):po.add(oi)
    po-=ap;pp-=ap;pp-=po
    return pL,pC,pR,po,pp

def build_optimal_attn(X_np,cw,stride,layer):
    T=len(X_np); opt=np.zeros((T,T))
    for pos in range(T):
        r=pos//stride;c=pos%stride
        if c>=cw:continue
        ac=(c+1)%cw;nr=r-1
        if nr<0:continue
        if layer==0:
            for dc in [-1,0,1]:
                nc=(ac+dc)%cw;pi=nr*stride+nc
                if pi<T:opt[pos,pi]=1./3
        elif layer==1:
            ncs=[(ac-1)%cw,ac,(ac+1)%cw]
            pb=[int(X_np[nr*stride+nc]) for nc in ncs];qp=tuple(pb)
            mps=[]
            for rr in range(0,r):
                for cc in range(cw):
                    l=(cc-1)%cw;ri=(cc+1)%cw
                    bits=(int(X_np[rr*stride+l]),int(X_np[rr*stride+cc]),int(X_np[rr*stride+ri]))
                    if bits==qp:
                        op=(rr+1)*stride+cc
                        if op<T and op<pos:mps.append(op)
            if mps:
                w=1./len(mps)
                for mp in mps:opt[pos,mp]=w
    return opt

# ============ t-SNE (two rows: by configuration, by output) ============
def make_fig_tsne(model,ds,dev,cw,ncr,ns=500,seed=42,od='.'):
    print("Generating t-SNE..."); np.random.seed(seed); stride=cw+1
    idx=np.random.choice(len(ds),min(ns,len(ds)),replace=False)
    sub=Subset(ds,idx)
    skeys=['embedding','L0_post_attn','L0_post_ffn','L1_post_attn','L1_post_ffn']
    stitles={'embedding':'Embedding','L0_post_attn':'After Layer 1\nattention',
             'L0_post_ffn':'After Layer 1\nMLP','L1_post_attn':'After Layer 2\nattention',
             'L1_post_ffn':'After Layer 2\nMLP'}
    sh={k:[] for k in skeys};ap=[];ao=[]
    ldr=DataLoader(sub,batch_size=min(64,len(sub)),shuffle=False)
    for bX,by,bm in ldr:
        outs=extract_layer_outputs(model,bX,dev)
        for b in range(bX.shape[0]):
            cv=bX[b].numpy();tv=by[b].numpy();T=len(cv)
            for pos in range(T):
                if bm[b,pos].item()==0:continue
                if pos%stride>=cw:continue
                if pos//stride<ncr:continue
                pat=get_parent_pattern(cv,pos,cw,stride)
                if pat<0:continue
                ap.append(pat);ao.append(int(tv[pos]))
                for k in skeys:sh[k].append(outs[k][b,pos])
    for k in skeys:sh[k]=np.array(sh[k])
    ap=np.array(ap);ao=np.array(ao)
    mx=4000
    if len(ap)>mx:ii=np.random.choice(len(ap),mx,replace=False)
    else:ii=np.arange(len(ap))
    tp=ap[ii];to_=ao[ii];pn=[f"{p:03b}" for p in range(8)]

    # Pre-compute t-SNE coordinates (shared across both rows)
    ncols = len(skeys)
    tsne_coords = {}
    for key in skeys:
        h=sh[key][ii]
        tsne=TSNE(n_components=2,perplexity=30,random_state=seed)
        tsne_coords[key]=tsne.fit_transform(h)

    # Combined figure: row 1 = by configuration, row 2 = by output
    fig,axes=plt.subplots(2,ncols,figsize=(1.6*ncols,3.6))

    # Row 1: colored by neighborhood configuration
    for si,key in enumerate(skeys):
        ax=axes[0,si]; coords=tsne_coords[key]
        for p in range(8):
            m=tp==p
            ax.scatter(coords[m,0],coords[m,1],c=TAB10[p],s=5,alpha=0.5,
                       edgecolors='none',zorder=2)
        ax.set_title(stitles[key],fontsize=6.5,pad=4);ax.set_xticks([]);ax.set_yticks([])
        for s in ax.spines.values():s.set_linewidth(0.3)
    axes[0,0].set_ylabel('By configuration',fontsize=7,labelpad=8)

    # Row 2: colored by output value
    out_colors={0:'#4E79A7',1:'#E15759'}
    for si,key in enumerate(skeys):
        ax=axes[1,si]; coords=tsne_coords[key]
        for v,c in out_colors.items():
            m=to_==v
            ax.scatter(coords[m,0],coords[m,1],c=c,s=5,alpha=0.5,
                       edgecolors='none',zorder=2)
        ax.set_xticks([]);ax.set_yticks([])
        for s in ax.spines.values():s.set_linewidth(0.3)
    axes[1,0].set_ylabel('By output',fontsize=7,labelpad=8)

    # Legends
    ch=[mpatches.Patch(facecolor=TAB10[p],label=pn[p]) for p in range(8)]
    oh=[mpatches.Patch(facecolor=out_colors[0],label='output 0'),
        mpatches.Patch(facecolor=out_colors[1],label='output 1')]
    plt.tight_layout()
    fig.legend(handles=ch,loc='lower center',ncol=8,fontsize=5,frameon=False,
               bbox_to_anchor=(0.45,-0.01),handlelength=1.0,handletextpad=0.3,columnspacing=0.5)
    fig.legend(handles=oh,loc='lower center',ncol=2,fontsize=5,frameon=False,
               bbox_to_anchor=(0.88,-0.01),handlelength=1.0,handletextpad=0.3,columnspacing=0.8)

    p=os.path.join(od,'fig_tsne')
    plt.savefig(p+'.pdf',bbox_inches='tight',pad_inches=0.02)
    plt.savefig(p+'.png',bbox_inches='tight',pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {p}.pdf/png")
    print(f"  t-SNE done ({len(ii)} points)")

# ============ Grid attention ============
def draw_grid_panel(ax,cv,h_attn,qfi,cw,stride,ncr,title,nb_set=None,mo_set=None,mp_set=None):
    T=len(cv);nrows=(T+stride-1)//stride
    qr=qfi//stride;qc=qfi%stride;ac=(qc+1)%cw
    vis=[r_*stride+c_ for r_ in range(nrows) for c_ in range(cw)
         if r_*stride+c_<qfi and r_*stride+c_<T]
    max_a=max([h_attn[i] for i in vis]) if vis else 1e-8
    cs=1.0  # square cells

    for r_ in range(nrows):
        for c_ in range(cw):
            flat=r_*stride+c_
            if flat>=T:continue
            x=c_*cs; y=(nrows-1-r_)*cs
            is_q=(r_==qr and c_==ac); is_f=flat>qfi
            a=h_attn[flat] if flat<len(h_attn) else 0
            t=min(a/max_a,1.0) if max_a>0 else 0

            if is_q:
                bg='#dbeafe';tc='#1e40af';dv='?'
            elif is_f:
                bg='#f5f5f5';tc='#ccc';dv=str(int(cv[flat]))
            else:
                rc_=int(247-t*215)/255;gc_=int(249-t*220)/255;bc_=int(253-t*120)/255
                bg=(rc_,gc_,bc_);tc='#fff' if t>0.5 else '#333';dv=str(int(cv[flat]))

            rect=mpatches.FancyBboxPatch((x+0.04,y+0.04),cs-0.08,cs-0.08,
                boxstyle="round,pad=0.01",facecolor=bg,edgecolor='#e0e0e0',linewidth=0.15)
            ax.add_patch(rect)

            # Borders for special cells
            if is_q:
                ax.add_patch(mpatches.FancyBboxPatch((x+0.02,y+0.02),cs-0.04,cs-0.04,
                    boxstyle="round,pad=0.01",facecolor='none',edgecolor=COLOR_QUERY,linewidth=1.5))
            elif nb_set and (r_,c_) in nb_set:
                ax.add_patch(mpatches.FancyBboxPatch((x+0.02,y+0.02),cs-0.04,cs-0.04,
                    boxstyle="round,pad=0.01",facecolor='none',edgecolor=COLOR_NB,linewidth=1.2))
            elif mo_set and (r_,c_) in mo_set:
                ax.add_patch(mpatches.FancyBboxPatch((x+0.02,y+0.02),cs-0.04,cs-0.04,
                    boxstyle="round,pad=0.01",facecolor='none',edgecolor=COLOR_MATCH_OUT,linewidth=1.2))
            elif mp_set and (r_,c_) in mp_set:
                ax.add_patch(mpatches.FancyBboxPatch((x+0.02,y+0.02),cs-0.04,cs-0.04,
                    boxstyle="round,pad=0.01",facecolor='none',edgecolor=COLOR_MATCH_PAR,linewidth=1.0))

            ax.text(x+cs/2,y+cs/2,dv,ha='center',va='center',fontsize=4.5,
                    fontweight='semibold',color=tc)

        # Row label
        ax.text(-0.3,(nrows-1-r_)*cs+cs/2,f't{r_}',ha='right',va='center',
                fontsize=4.5,color='#999')

    ax.set_xlim(-0.5,cw*cs+0.2)
    ax.set_ylim(-0.2,nrows*cs+0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title,fontsize=8,pad=4)

def make_fig_grid_attn(model,ds,dev,cw,ncr,ng=20,seed=42,od='.'):
    print(f"Generating {ng} grid attention figures...")
    np.random.seed(seed); stride=cw+1
    idx=np.random.choice(len(ds),min(1000,len(ds)),replace=False)
    good=[]
    model.eval()
    with torch.no_grad():
        for i in idx:
            X,y,mask=ds[i]
            out=model(X.unsqueeze(0).to(dev))
            pr=out[0].argmax(dim=-1).cpu()
            if not((pr!=y)&(mask>0)).any():good.append(i)
            if len(good)>=ng:break

    for gi,si in enumerate(good):
        X,y,mask=ds[si]
        # Append the last cell from target to complete the grid
        last_cell = y[-1:].numpy()
        cv=np.concatenate([X.numpy(), last_cell])
        T=len(cv)
        eps=[p for p in range(len(mask)) if mask[p].item()>0 and p%stride<cw]
        if not eps:continue
        qpos=eps[len(eps)//2]
        qr=qpos//stride;qc=qpos%stride;ac=(qc+1)%cw

        attn_maps,preds=extract_attention_maps(model,X.unsqueeze(0),dev)

        # Build annotation sets
        nr_=qr-1;nbc=[(ac-1)%cw,ac,(ac+1)%cw]
        nb_set=set();qpat=None;mo_set=set();mp_set=set()
        if nr_>=0:
            for c in nbc:nb_set.add((nr_,c))
            pb=[int(cv[nr_*stride+c]) for c in nbc];qpat=tuple(pb)
            for r in range(0,nr_):
                for c in range(cw):
                    l=(c-1)%cw;ri=(c+1)%cw
                    if r*stride+ri>=T:continue
                    bits=(int(cv[r*stride+l]),int(cv[r*stride+c]),int(cv[r*stride+ri]))
                    if bits==qpat:
                        mp_set.update([(r,l),(r,c),(r,ri)])
                        mo_set.add((r+1,c))
            mo_set-=nb_set;mp_set-=nb_set;mp_set-=mo_set

        # Horizontal layout: Layer 1 | Layer 2 (average across heads per layer)
        fig,axes=plt.subplots(1,2,figsize=(7.0,3.5))

        for li in range(2):
            if li not in attn_maps:continue
            attn=attn_maps[li]
            ha=attn.mean(axis=0)[qpos,:]  # average across heads
            tl='Layer 1' if li==0 else 'Layer 2'
            draw_grid_panel(axes[li],cv,ha,qpos,cw,stride,ncr,tl,
                           nb_set,mo_set,mp_set)

        # Legend (closer to figure)
        lh=[mpatches.Patch(facecolor='#dbeafe',edgecolor=COLOR_QUERY,linewidth=1.5,label='Query cell'),
            mpatches.Patch(facecolor='none',edgecolor=COLOR_NB,linewidth=1.2,label='Neighborhood'),
            mpatches.Patch(facecolor='none',edgecolor=COLOR_MATCH_OUT,linewidth=1.2,label='Matched output'),
            mpatches.Patch(facecolor='none',edgecolor=COLOR_MATCH_PAR,linewidth=1.0,label='Matched neighborhood')]
        plt.tight_layout()
        # Add legend right below the axes
        fig.legend(handles=lh,loc='lower center',ncol=4,fontsize=6,
                   frameon=False,bbox_to_anchor=(0.5,-0.01))

        # Filename encodes sample info + rule number
        tv=int(y[qpos].item());pv=int(preds[qpos])
        ps="".join(str(b) for b in pb) if qpat else "x"
        rule_num=int(ds.rules[si]) if ds.rules is not None else -1
        fname=f'fig_grid_{gi:02d}_s{si}_rule{rule_num}_t{qr}c{ac}_pat{ps}_pred{pv}_true{tv}'
        p=os.path.join(od,fname)
        plt.savefig(p+'.pdf',bbox_inches='tight',pad_inches=0.02)
        plt.savefig(p+'.png',bbox_inches='tight',pad_inches=0.02)
        plt.close(fig)
        print(f"  Saved {fname}.pdf")

# ============ Heatmaps ============
def make_fig_heatmap(model,ds,dev,cw,ncr,ns=100,od='.'):
    print("Generating heatmap...");stride=cw+1
    idx=np.random.choice(len(ds),min(ns,len(ds)),replace=False)
    s0=ds[0];T=len(s0[0]);nl=len(list(model.transformer_layers))
    aa={li:None for li in range(nl)};ao1=np.zeros((T,T));cnt=0
    for i in idx:
        X,y,m=ds[i]
        am,_=extract_attention_maps(model,X.unsqueeze(0),dev)
        ol2=build_optimal_attn(X.numpy(),cw,stride,layer=1)
        for li in range(nl):
            a=am[li].mean(axis=0)
            if aa[li] is None:aa[li]=np.zeros_like(a)
            aa[li]+=a
        ao1+=ol2;cnt+=1
    for li in aa:aa[li]/=cnt
    ao1/=cnt
    nrt=(T+stride-1)//stride;rc=[r*stride+cw/2 for r in range(nrt)]
    def dh(ax,at,cm,tl):
        im=ax.imshow(at,cmap=cm,aspect='equal',interpolation='nearest')
        for r in range(1,nrt):
            sp=r*stride-0.5
            ax.axhline(y=sp,color='white',linewidth=0.3,alpha=0.7)
            ax.axvline(x=sp,color='white',linewidth=0.3,alpha=0.7)
        ax.set_xticks(rc);ax.set_xticklabels([f't{r}' for r in range(nrt)],fontsize=5)
        ax.set_yticks(rc);ax.set_yticklabels([f't{r}' for r in range(nrt)],fontsize=5)
        ax.set_xlabel('Key',fontsize=6);ax.set_ylabel('Query',fontsize=6)
        ax.set_title(tl,fontsize=7,pad=4)
        div=make_axes_locatable(ax);cax=div.append_axes("right",size="4%",pad=0.05)
        plt.colorbar(im,cax=cax).ax.tick_params(labelsize=4)
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.5))
    dh(axes[0],aa[0],CMAP_L1,'Layer 1 (learned)')
    dh(axes[1],aa[1],CMAP_L2,'Layer 2 (learned)')
    dh(axes[2],ao1,CMAP_OPT,'Layer 2 (optimal)')
    plt.tight_layout()
    p=os.path.join(od,'fig_heatmap');plt.savefig(p+'.pdf');plt.savefig(p+'.png');plt.close(fig)
    print(f"  Saved {p}.pdf/png ({cnt} samples)")

# ============ Attention distribution ============
BC1={'left':'#6BAED6','center':'#2171B5','right':'#6BAED6','total':'#08519C',
     'prior_output':'#9ECAE1','prior_parent':'#C6DBEF','other':'#DEEBF7'}
BC2={'left':'#74C476','center':'#238B45','right':'#74C476','total':'#006D2C',
     'prior_output':'#00441B','prior_parent':'#A1D99B','other':'#C7E9C0'}

def make_fig_attn_dist(model,ds,dev,cw,ncr,ns=0,seed=42,od='.'):
    print("Generating attn dist...");np.random.seed(seed);stride=cw+1
    if ns<=0 or ns>=len(ds):
        sub=ds; print(f"  Using all {len(ds)} samples")
    else:
        idx=np.random.choice(len(ds),ns,replace=False);sub=Subset(ds,idx)
        print(f"  Using {ns} samples")
    cats=['left','center','right','prior_output','prior_parent','other']
    ls=defaultdict(lambda:defaultdict(float));lc=defaultdict(int)
    for i,(X,y,m) in enumerate(sub):
        if(i+1)%500==0:print(f"  {i+1}/{len(sub)}...")
        am,_=extract_attention_maps(model,X.unsqueeze(0),dev);cv=X.numpy();T=len(cv)
        for pos in range(T):
            if m[pos].item()==0:continue
            if pos%stride>=cw:continue
            if pos//stride<ncr:continue
            pL,pC,pR,po,pp=get_cell_categories_detailed(cv,pos,cw,stride)
            if not pL:continue
            for li in am:
                ha=am[li].mean(axis=0)[pos,:]
                la=sum(ha[j] for j in pL);ca=sum(ha[j] for j in pC);ra=sum(ha[j] for j in pR)
                poa=sum(ha[j] for j in po);ppa=sum(ha[j] for j in pp)
                oa=1.0-la-ca-ra-poa-ppa
                ls[li]['left']+=la;ls[li]['center']+=ca;ls[li]['right']+=ra
                ls[li]['prior_output']+=poa;ls[li]['prior_parent']+=ppa
                ls[li]['other']+=oa;lc[li]+=1
    la={}
    for li in sorted(ls.keys()):
        la[li]={c:ls[li][c]/lc[li] for c in cats}
        la[li]['total_neighbor']=la[li]['left']+la[li]['center']+la[li]['right']
    for li in sorted(la.keys()):
        a=la[li]
        print(f"  L{li}: L={a['left']:.4f} C={a['center']:.4f} R={a['right']:.4f} "
              f"total={a['total_neighbor']:.4f} | po={a['prior_output']:.4f} "
              f"pp={a['prior_parent']:.4f} oth={a['other']:.4f}")

    fig,axes=plt.subplots(1,2,figsize=(7.0,2.2))
    for li,ax in enumerate(axes):
        if li not in la:continue
        a=la[li];bc=BC1 if li==0 else BC2
        labs=['Neighbor (L)','Neighbor (C)','Neighbor (R)','',
              'Total neighbor','',
              'Prior output','Prior parent','Other']
        vals=[a['left'],a['center'],a['right'],0,
              a['total_neighbor'],0,
              a['prior_output'],a['prior_parent'],a['other']]
        cols=[bc['left'],bc['center'],bc['right'],'none',
              bc['total'],'none',
              bc['prior_output'],bc['prior_parent'],bc['other']]
        yp=list(range(len(labs)));yp.reverse()
        ax.barh(yp,vals,color=cols,height=0.7,edgecolor='none')
        ax.set_yticks(yp);ax.set_yticklabels(labs,fontsize=5.5)
        ax.set_xlim(0,1.08);ax.set_xlabel('Attention weight',fontsize=6)
        ax.set_title(f'Layer {li+1}',fontsize=8,pad=4)
        for y_,v in zip(yp,vals):
            if v>0.001:ax.text(v+0.01,y_,f'{v:.3f}',va='center',fontsize=4.5,color='#666')
        ax.axhline(y=yp[3]+0.5,color='#ddd',linewidth=0.5)
        ax.axhline(y=yp[5]+0.5,color='#ddd',linewidth=0.5)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y',length=0)
    plt.tight_layout()
    p=os.path.join(od,'fig_attn_dist');plt.savefig(p+'.pdf');plt.savefig(p+'.png');plt.close(fig)
    print(f"  Saved {p}.pdf/png")

# ============ Main ============
def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--ckpt",type=str,required=True)
    pa.add_argument("--test_path",type=str,required=True)
    pa.add_argument("--cell_width",type=int,required=True)
    pa.add_argument("--num_context_rows",type=int,default=4)
    pa.add_argument("--output_dir",type=str,default="./figures")
    pa.add_argument("--num_tsne_samples",type=int,default=500)
    pa.add_argument("--num_heatmap_samples",type=int,default=100)
    pa.add_argument("--num_dist_samples",type=int,default=0,
                    help="0 = use all samples")
    pa.add_argument("--num_grids",type=int,default=20)
    pa.add_argument("--seed",type=int,default=42)
    pa.add_argument("--skip_tsne",action="store_true")
    pa.add_argument("--skip_grid",action="store_true")
    pa.add_argument("--skip_heatmap",action="store_true")
    pa.add_argument("--skip_dist",action="store_true")
    args=pa.parse_args()
    random.seed(args.seed);np.random.seed(args.seed)
    os.makedirs(args.output_dir,exist_ok=True)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading: {args.ckpt}")
    model,cfg=load_model(args.ckpt,dev)
    print(f"Config: heads={cfg['heads_list']}, d={cfg['hidden_size']}")
    ds=ECADataset(args.test_path);print(f"Test: {len(ds)} samples\n")
    if not args.skip_tsne:
        make_fig_tsne(model,ds,dev,args.cell_width,args.num_context_rows,
                      args.num_tsne_samples,args.seed,args.output_dir)
    if not args.skip_grid:
        make_fig_grid_attn(model,ds,dev,args.cell_width,args.num_context_rows,
                           args.num_grids,args.seed,args.output_dir)
    if not args.skip_heatmap:
        make_fig_heatmap(model,ds,dev,args.cell_width,args.num_context_rows,
                         args.num_heatmap_samples,args.output_dir)
    if not args.skip_dist:
        make_fig_attn_dist(model,ds,dev,args.cell_width,args.num_context_rows,
                           args.num_dist_samples,args.seed,args.output_dir)
    print(f"\nDone! All in {args.output_dir}/")

if __name__=="__main__":main()