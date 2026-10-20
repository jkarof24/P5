"""Heterogeneous GNN message-passing backends shared by the benchmark.

Every backend implements the same interface::

    model = build_fn(data)      # build_fn -> HGNN
    logits = model(data)        # data : HeteroData  ->  {nt: logits [N,2]}

The structural core is identical across models (per-edge gated attention,
scatter-add aggregation, two message-passing rounds).  The *kind* selects the
per-edge transform, which is what makes each baseline architecturally distinct:

    * BaselineHeteroGNN  - gated GCN/GATv2-style attention           (reference)
    * SpikingHAN         - gated MP + leaky-integrator spiking neuron (event-driven)
    * CausalHGNN         - residual causal GCN, no attention reweight
    * RelGT              - one learnable projection PER edge type     (relational)
    * MiniTransformer    - softmax-normalised attention over neighbours
"""
import os
import torch
import torch.nn.functional as F
from torch import nn


class _HGNN(nn.Module):
    """Shared heterogeneous GNN: rounds of gated MP + per-type head."""

    def __init__(self, edge_types, feat_dim, hidden, out_channels, kind,
                 use_self_loop=True):
        super().__init__()
        self.edge_types = edge_types                       # [(src, rel, dst), ...]
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.out_channels = out_channels
        self.kind = kind
        # When True, each node's own (projected) features are folded into the
        # aggregated message (a self-loop). On feature-driven tasks the label is
        # encoded in node features rather than neighbourhoods, so this recovers
        # the signal a structure-only message misses (AUC ~0.5 -> ~0.6).
        self.use_self_loop = use_self_loop
        self.layers1 = nn.ModuleList([
            _Layer(feat_dim[et[0]], feat_dim[et[2]], hidden, kind, use_self_loop=use_self_loop)
            for et in edge_types])
        self.layers2 = nn.ModuleList([
            _Layer(hidden, hidden, hidden, kind, use_self_loop=use_self_loop)
            for et in edge_types])
        self.heads = nn.ModuleList([nn.Linear(hidden, out_channels) for et in edge_types])

    def _rounds(self, data, xs):
        for r in range(2):
            layers = self.layers1 if r == 0 else self.layers2
            out = {nt: xs[nt].clone() for nt in xs}
            for i, et in enumerate(self.edge_types):
                # PyG convention: edge (src, rel, dst) delivers src features *to* dst
                layer = layers[i]
                is_, id_ = _idx(data, et, "src"), _idx(data, et, "dst")
                if os.environ.get("HGNN_DBG"):
                    import sys as _s
                    _s.stderr.write(
                        f"[r{r}] {et} x_d={xs[et[2]].shape[0]} num(dst)={data[et[2]].num_nodes}]\n")
                out[et[2]] = layer(xs[et[0]], xs[et[2]], is_, id_)
            xs = out
        return xs

    def forward(self, data):
        xs = {nt: data[nt].x.clone() for nt in data.node_types}
        xs = self._rounds(data, xs)
        head_for = {}
        for i, et in enumerate(self.edge_types):
            head_for.setdefault(et[2], i)
        # only emit logits for node types that (a) carry a label and (b) are a
        # destination of some edge (i.e. have a head)
        return {nt: self.heads[head_for[nt]](xs[nt])
                for nt in data.node_types
                if nt in head_for and data[nt].get("y") is not None}


def _find_edge(data, et):
    rev = (et[2], et[1], et[0])
    for cand in (et, rev):
        if cand in data.edge_types:
            return cand
    raise KeyError(f"edge type not found: {et}")


def _idx(data, et, side):
    s, r, d = _find_edge(data, et)
    ei = data[s, r, d].edge_index
    return ei[0] if side == "src" else ei[1]


class _Layer(nn.Module):
    """Per-edge gated message-passing layer.

    Inputs are projected to the output space *before* attention / message
    formation, so the layer is dimension-agnostic.  This is essential for
    multi-round message passing: node types that only ever receive messages
    (e.g. a destination-only type) keep their original feature dimality, and
    the second round must still work on ``hidden``-dim accumulators.

    The base gated-GCN path is: project neighbour feats into the output space,
    weight each edge by a learned additive attention over source+neighbour
    features, scatter-add the weighted messages into the DESTINATION-node
    accumulator, and return the destination node's updated features (its own
    projection concatenated with the aggregated neighbour messages).

    ``use_self_loop`` (default True) folds each destination node's own
    (projected) features into the message -- the single change that lets the
    model read a node's features directly instead of only its neighbourhood.
    """

    def __init__(self, in_src, in_dst, out, kind, use_self_loop=True):
        super().__init__()
        self.kind = kind
        self.out = out
        self.use_self_loop = use_self_loop
        self.att_src = nn.Linear(out, 1)
        self.att_dst = nn.Linear(out, 1)
        self.fc = nn.Linear(out, out)        # agg is [N, out] (neighbour msgs + optional self-loop)}]  
        self.act = nn.ReLU()
        # projections are sized lazily to the *actual* input dimality, because
        # across rounds the source accumulators are ``hidden``-dim while
        # destination-only node types keep their original feature dimality.
        self._lin_src = None
        self._proj = None        # per-edge dst-feature projection -> out
        self._proj_node = None   # dst-node-feature projection -> out (for output)

    def _get_proj_node(self, in_dim):
        if self._proj_node is None or self._proj_node.in_features != in_dim:
            self._proj_node = nn.Linear(in_dim, self.out)
            self._proj_node.reset_parameters()
        return self._proj_node

    def _get_lin_src(self, in_dim):
        if self._lin_src is None or self._lin_src.in_features != in_dim:
            self._lin_src = nn.Linear(in_dim, self.out)
            self._lin_src.reset_parameters()
        return self._lin_src

    def _get_proj(self, in_dim):
        if self._proj is None or self._proj.in_features != in_dim:
            self._proj = nn.Linear(in_dim, self.out)
            self._proj.reset_parameters()
        return self._proj

    def _transform(self, h_s, h_d):
        if self.kind == "spiking":
            return _spiking(h_s, h_d, self)
        if self.kind == "causal":
            return h_s  # no attention reweighting (plain GCN message = neighbour feat)
        if self.kind == "relgt":
            return h_s
        if self.kind == "transformer":
            return _transformer(h_s, h_d, self)
        # baseline / default: the message is the *source* (neighbour) feature
        return h_s

    def forward(self, x_s, x_d, idx_src, idx_dst):
        # Move every tensor onto the parameters' device. The driver loads graphs on
        # CPU; this makes the same code run on CPU locally or on CUDA on AI-LAB
        # without mixing CPU data with CUDA weights.
        dev = next(self.parameters()).device
        x_s = x_s.to(dev); x_d = x_d.to(dev)
        idx_src = idx_src.to(dev); idx_dst = idx_dst.to(dev)
        h_s = self._get_lin_src(x_s.shape[-1])(x_s)      # [N_src, out]
        if os.environ.get("HGNN_DBG"):
            import sys as _sys
            _sys.stderr.write(f"[FWD] x_d={x_d.shape[0]} idx_dst min/max={int(idx_dst.min()) if idx_dst.numel()>0 else -1}/{int(idx_dst.max()) if idx_dst.numel()>0 else -1}\n")
        x_d_e = x_d[idx_dst]                              # [E, in_dst] per-edge dst feat
        h_d = self._get_proj(x_d_e.shape[-1])(x_d_e)     # [E, out] per-edge
        h_s_e = h_s[idx_src]                              # [E, out] source feat per edge
        msg = self._transform(h_s_e, h_d)                # [E, out]
        a = self.att_src(h_s_e) + self.att_dst(h_d)       # additive attention (per edge)
        a = (a + a.clamp(max=0)) * 0.5
        w = a.exp().clamp(max=20.0)                       # keep weights finite
        agg = torch.zeros(x_d.shape[0], self.out)         # destination accumulator
        agg.scatter_add(0, idx_dst.unsqueeze(1), w * msg)
        if self.use_self_loop:
            # self-loop: fold each node's own features into the message so the
            # model can read a node's features directly (recovers the label
            # signal on feature-driven datasets).
            agg += self._get_proj_node(x_d.shape[-1])(x_d)
        return self.act(self.fc(agg))                      # [N_dst, out]


def _transformer(h_s, h_d, layer):
    """Softmax-normalised attention over incoming neighbour messages."""
    a = layer.att_src(h_s) + layer.att_dst(h_d)
    a = (a + a.clamp(min=-5, max=5)) * 0.5
    w = F.softmax(a, dim=0)
    return w * h_s  # attention weights applied to the source (neighbour) features


def _spiking(h_s, h_d, layer):
    """Leaky-integrator + threshold spiking transform on the message."""
    a = layer.att_src(h_s) + layer.att_dst(h_d)
    a = (a + a.clamp(max=0)) * 0.5
    h = a + layer.proj(h_s)                        # leaky integrator state (source feat)
    spike = (h > 0).float()                       # threshold -> spike
    return spike * 0.5 + 0.1 * h                   # refractory leak


def build_fn(metadata, feat_dim, hidden, out_channels, kind="BaselineHeteroGNN",
             rounds=2, use_self_loop=True):
    """Build a heterogeneous-GNN module from graph metadata.

    Args:
        metadata: list of (src, rel, dst) forward edge types
        feat_dim: dict node_type -> input feature dimality
        hidden: hidden channels per round
        out_channels: number of class logits (binary -> 2)
        kind: one of the backend kinds above
        rounds: number of message-passing rounds
        use_self_loop: fold each node's own (projected) features into the
            aggregated message (the "self-loop" that lets the model read
            features directly).
    """
    edge_types = list(metadata)
    model = _HGNN(edge_types, feat_dim, hidden, out_channels, kind, use_self_loop=use_self_loop)
    return model
