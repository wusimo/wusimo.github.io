---
layout: post
title: 'Attention Residuals: Making the "+" Learned'
date: 2026-05-17 10:00:00-0400
description: 'Part 2 — replacing the fixed `+` with a softmax over prior block outputs.'
tags: transformers attention adaptive-computation
categories: ml
related_posts: true
---

*Part 2 of a series on attention residuals and AR-Retrofit. [Part 1](/blog/2026/attention-and-residuals/) set up the residual stream and the question this post answers.*

## Picking up from Post 1

We ended the last post with two questions about the residual `+`:

- Why should block $$l$$ only see the immediately previous state $$x_{l-1}$$? Every earlier state $$x_0, x_1, \ldots, x_{l-1}$$ is already computed and sitting in the residual stream.
- Why should the combination weight be a fixed scalar 1, the same for every block, every layer, every input?

The natural reaction once you see those questions is: *attention*. The whole point of attention is to let a query selectively combine multiple sources. We already use it across tokens. Why not across depth?

That is the move in **Attention Residuals** (Kimi Team, 2026). This post walks through what changes when you make it.

## Attention residuals in one page

Recall the standard residual recipe:

$$
x_l = x_{l-1} + f_l(\mathrm{LN}(x_{l-1}))
$$

where $$f_l$$ is the $$l$$-th block (MHA + FFN, in pre-norm form). The `+` is a fixed scalar-1 combination of two things: the previous state $$x_{l-1}$$, and the block's contribution $$f_l(\cdot)$$.

In **Attention Residuals (AttnRes)**, that combination becomes a learned softmax-weighted sum over *all* previous block outputs.

Let $$h_i$$ denote the output of block $$i$$ — what would have been added to the residual stream in the standard recipe. AttnRes replaces the `+` with:

$$
\begin{aligned}
k_i &= W_K\, h_i \\
\alpha_{i \to l} &= \mathrm{softmax}_i\!\left(\frac{w_l^\top k_i}{\sqrt d}\right), \quad i = 0, 1, \ldots, l-1 \\
x_l &= \sum_{i=0}^{l-1} \alpha_{i \to l}\, h_i
\end{aligned}
$$

Three new objects show up:

- $$w_l \in \mathbb{R}^d$$ is a **learned pseudo-query** owned by block $$l$$. There is one per block; no per-token query.
- $$k_i = W_K h_i$$ are **keys** projected from each prior block output. $$W_K$$ is shared across blocks.
- $$\alpha_{i \to l}$$ is a softmax over depth — for each layer $$l$$, a distribution over the $$l$$ prior states.

The result $$x_l$$ is the input to block $$l$$. The block itself ($$f_l$$) is unchanged: it is still MHA + FFN, still operating on a $$d$$-dimensional input. What changed is *how that input is assembled from the history of the stream*.

```text
def attention_residual(history, l):
    # history = [h_0, h_1, ..., h_{l-1}]
    K = stack([W_K @ h for h in history])      # (l, d)
    scores  = (w[l] @ K.T) / sqrt(d)           # (l,)
    alpha   = softmax(scores)                  # (l,)
    x_l     = sum(alpha[i] * history[i] for i in range(l))
    return x_l
```

That is the entire mechanism. No per-token routing; no extra normalization layers; one query vector per block, one shared key projection.

<p align="center"><img src="/assets/img/blog/diagram3_attnres_fanin.svg" alt="Attention Residual fan-in" width="620"></p>

*Diagram 3. Block $$l$$'s input is no longer a single `+` from $$x_{l-1}$$ — it is a softmax-weighted fan-in from every prior output $$h_0, \ldots, h_{l-1}$$. The router (dashed) computes $$\alpha$$ before block $$l$$ runs; edge thickness illustrates the weight magnitude.*

## Recovering the standard residual

A useful sanity check: AttnRes is a strict generalization of the standard `+`. If we constrain $$\alpha$$ to put all its mass on $$i = l-1$$, we recover

$$
x_l = h_{l-1} = x_{l-1} + f_{l-1}(\mathrm{LN}(x_{l-1}))
$$

which is the standard recipe (with $$h_{l-1}$$ playing the role of the post-residual state, depending on whether you define $$h$$ as block-output-only or full-state — Kimi's paper uses the latter). Initializing $$w_l$$ to peak on $$i = l-1$$, or warming up from a one-hot at $$i=l-1$$ toward learned weights, gives an identity-at-initialization that mirrors how standard residuals already start. This will matter a lot in Post 3.

## Three views of attention residuals

The fixed `+` of Post 1 was doing three things at once: enabling gradient flow, providing identity-at-init, and acting as a working-memory write into a shared stream. The learned version inherits all three — and then adds three more capabilities of its own.

### 1. Input-dependent depth combination

In the standard residual, every input traverses the same path: every block contributes with the same weight 1. The depth of computation is fixed by the architecture.

With AttnRes, different inputs induce different distributions over $$\alpha$$. An easy token (say, completing a common bigram) might end up with $$\alpha$$ concentrated on shallow $$i$$ — the model is signaling that it does not need deep blocks. A hard token (say, multi-hop reasoning) might spread $$\alpha$$ across deep blocks, or weight a specific mid-stack block heavily.

This is *adaptive computation*, but not in the explicit halting sense of [ACT](https://arxiv.org/abs/1603.08983) or the per-token routing of [MoD](https://arxiv.org/abs/2404.02258). It is a softer thing: the *combination* over depth is input-dependent, even though every block still runs. The model decides which prior states matter for the current block, not whether to skip.

### 2. A routing signal that exists *before* the block runs

This is the property that turns out to be most useful, and it is easy to miss.

Look again at the order of operations:

1. Compute $$\alpha_{i \to l}$$ from prior outputs $$h_0, \ldots, h_{l-1}$$ and the learned $$w_l$$.
2. Form $$x_l = \sum_i \alpha_{i \to l} h_i$$.
3. Run block $$l$$ on $$x_l$$ to produce $$h_l$$.

Step 1 finishes before step 3 starts. The weights $$\alpha_{\cdot \to l}$$ are a *pre-execution* signal about how much block $$l$$'s input is drawing on each predecessor — and, by symmetry, how much *future* blocks will draw on $$h_l$$ once it exists.

This is the foothold that ReSkip (Post 3) uses to decide which blocks to skip at inference. If we can read off "block $$l$$ is barely contributing to the stream" before block $$l$$ runs, we can skip it. Standard residuals give us no such signal: every block contributes with weight 1 by definition.

### 3. Depth as a soft mixture, not a strict ordering

The standard residual stream is sequential by construction: block $$l$$ reads only what block $$l-1$$ wrote (plus everything that survived the identity path). AttnRes lets block $$l$$ read directly from $$h_3$$ or $$h_7$$ without going through $$h_{l-1}$$.

This breaks the implicit assumption that information has to *survive* intermediate writes to reach a later block. In standard transformers, a feature written by block 3 only reaches block 15 if blocks 4 through 14 do not stomp on it. AttnRes lets block 15 reach back to $$h_3$$ directly. The residual stream stops being a strictly ordered tape and becomes more like a content-addressable memory indexed by depth.

The Kimi paper finds that this matters most at moderate depths and with block-level (not layer-level) granularity — grouping $$S \in \{2, 4, 8\}$$ layers into a block before applying AttnRes. Layer-level granularity is too fine; the router has nothing useful to say between adjacent layers.

## What is hiding in $$w_l$$

The learned pseudo-query $$w_l$$ is the new degree of freedom, and it deserves a beat.

In standard attention, queries are per-token: each position computes its own query from its hidden state. In AttnRes, the query is per-block: it is a single vector, learned once, that does not depend on the input. The keys are input-dependent (they come from $$h_i$$, which depends on the input), but the query is not.

That seems like a strong restriction, and it is — but it is also what makes the mechanism cheap and what makes the routing weights *interpretable as a property of the block*, not of the token. "Block $$l$$ tends to read from blocks 3, 7, and $$l-1$$" is a statement about block $$l$$. If $$w_l$$ were per-token, we would have to talk about the routing of each (block, token) pair separately, and the signal would be much noisier.

This is also why AttnRes is comparatively cheap. The added parameters per block are $$w_l \in \mathbb{R}^d$$ and a shared $$W_K \in \mathbb{R}^{d \times d}$$ — that is one vector per block plus one matrix total, against the millions of parameters in the block itself.

<p align="center"><img src="/assets/img/blog/diagram4_alpha_heatmap.png" alt="Heatmap of routing weights across depth" width="560"></p>

*Diagram 4. A heatmap of $$\alpha_{i \to l}$$ across depth (synthetic, illustrative). Rows are queries (block $$l$$), columns are predecessors $$i$$. The standard residual would be a strict subdiagonal one-hot; AttnRes shows a band along the subdiagonal plus a few off-diagonal hotspots where late blocks reach back to early ones.*

## Looking ahead

AttnRes gives us two new things compared to the standard `+`:

1. The depth combination is **input-dependent** and **learned**.
2. The routing weights $$\alpha$$ are **available before each block runs**, which makes them usable as a control signal.

There is one large catch: the recipe above assumes we are training from scratch. Every existing pretrained decoder — Qwen, Llama, the VLM backbones we actually deploy — was trained with the fixed `+`. Their weights are entangled with the assumption that block $$l$$ reads from $$x_{l-1}$$, and nothing else.

The question for Part 3 is the practical one: **can we retrofit attention residuals into a frozen pretrained model?** Without retraining the backbone, ideally with under half a percent of new parameters, and ideally in a way that lets us use the routing signal to skip blocks at inference. That is AR-Retrofit and ReSkip, and the answer is yes — with some design choices that turn out to matter.
