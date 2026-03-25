# Next Steps: BCA as Standard Baseline + r_info Fair Test

## Core Principle

BCA is not an experimental treatment — it is the **minimum capability** that any bridge bidding agent should possess. An agent that cannot interpret bidding history is not "playing bridge" in any meaningful sense. Therefore:

- **ALL agents use BCA** (belief-conditioned actor with partner + RHO inference, 397-dim)
- Actor input = base obs (301) + partner belief (48) + RHO belief (48) = **397-dim**
- The experimental variable is **r_info only**
- Convention drift quantification is a **separate, independent contribution**

**Why RHO (Right-Hand Opponent)?** In bridge, the player who bid immediately before you has the most direct impact on your decision (pass/overcall/double). LHO acts after you, so pre-inference on LHO has limited value. This mirrors real bridge cognition: you always process RHO's bid before deciding.

---

## Part 1: Code Modifications

### 1.1 No changes needed to BeliefNetwork architecture

`belief_net.py` already supports arbitrary `(observer_pos, target_pos)` pairs. The architecture is general — the limitation is purely in how it's called.

### 1.2 Modify `_get_belief_features_single()` in `subgame_trainer.py`

**Current behavior (partner-only, 48-dim):**
```python
partner = (player + 2) % 4
probs = bn.get_probs(oh_t, h_t, op_t=[player], tp_t=[partner])  # (1, 48)
```

**Change: query both partner AND RHO, return 96-dim.**
```python
partner = (player + 2) % 4
rho = (player - 1) % 4       # Right-Hand Opponent (bid just before you)

partner_probs = bn.get_probs(oh_t, h_t, op_t=[player], tp_t=[partner])  # (1, 48)
rho_probs     = bn.get_probs(oh_t, h_t, op_t=[player], tp_t=[rho])      # (1, 48)
return concat(partner_probs, rho_probs)  # (1, 96)
```

**Why RHO and not opponent-combined?**
- Partner belief tells you "partner probably has X" → you know 13 cards yourself → 39 unknown among 3 others
- P(card in opponent_combined) = 1 - P(card in partner) → mathematically redundant
- But P(card in RHO specifically) ≠ 1 - P(card in partner) → **this IS new information**
- RHO's bid (e.g. East's 1S) reveals RHO-specific distribution that partner belief cannot provide

Same change needed for `_get_belief_features_batch()`.

### 1.3 Modify `sl_pretrain_bca.py` Stage A: Add opponent-target training samples

**Current:** Only collects `(observer=player, target=partner)` samples.

**Change:** Also collect `(observer=player, target=LHO)` and `(observer=player, target=RHO)` samples. This teaches the Belief Network to predict any player's hand, not just partner's. This is needed for:
- r_info's opponent term (I(bid; hand | opponent))
- Future convention-sharing evaluation (opponent uses your belief net to interpret your bids)

```python
# In collect_all_data_from_sayc(), after the existing partner sample:
for target in [partner, (player+1)%4, (player+3)%4]:  # partner, LHO, RHO
    samples.append({
        'observer_hand':   hands[player].copy(),
        'history':         hist_enc,
        'observer_pos':    player,
        'target_pos':      target,
        'target_features': hand_to_belief_target(hands[target]),
    })
```

Training samples triple in size, but BeliefNetwork architecture stays identical. The position embeddings handle the rest.

### 1.4 Modify `subgame_validation.py`: All agents use BCA

**Current:** Agent A uses `belief_conditioned=False`, Agent B uses `belief_conditioned=True`.

**Change:** Both Agent A and Agent B use `belief_conditioned=True`. The only difference is r_info.

```python
# Agent A: MAPPO + BCA (control)
agent_a_config = SubgameConfig(
    belief_conditioned=True,   # ← was False
    beta=0.0,
    info_reward_weight=0.0,    # no r_info
    kl_lambda_start=0.3,
    ...
)

# Agent B: MAPPO + BCA + r_info (treatment)
agent_b_config = SubgameConfig(
    belief_conditioned=True,
    beta=0.0,                  # partner-only r_info (β=0)
    info_reward_weight=0.2,    # r_info active
    kl_lambda_start=0.3,
    ...
)
```

### 1.5 SL Checkpoint: Single unified checkpoint

Since all agents use BCA with partner + RHO belief, there is only ONE SL checkpoint: `sl_base_bca_v2.pt` (397-dim). No need for separate 301-dim or 349-dim checkpoints.

**`policy_net.py` constants update:**
```python
BELIEF_FEAT_DIM = 96           # 48 (partner) + 48 (RHO)
BELIEF_OBS_DIM  = 301 + 96     # = 397
```

`make_belief_features_prior()` returns 96-dim (two concatenated 48-dim priors).

### 1.6 FSP Pool: Minimal change

FSP opponents continue to use actor-only snapshots (no belief net stored). The only change is that `make_belief_features_prior()` now returns 96-dim (two concatenated 48-dim priors) to match the 397-dim actor input.

This is acceptable because:
- FSP opponents already have BCA capability baked into their actor weights (from SL pretraining on 397-dim inputs)
- They use `make_belief_features_prior()` at inference, which degrades gracefully (uniform prior for both partner and RHO)
- Fixing this properly (storing belief net in FSP) is significant engineering for marginal gain
- Document as known limitation: FSP opponents do not actively interpret our bids during training

---

## Part 2: Experiment Design

### 2.1 Experiment Matrix (Competitive Subgame 1H-1S)

| Agent | BCA | r_info | β | Role |
|-------|-----|--------|---|------|
| SL baseline | ✓ | ✗ | — | Reference convention (zero-drift anchor) |
| A: MAPPO+BCA | ✓ | ✗ | — | Does RL improve over SL when agents can "understand" bids? |
| B: MAPPO+BCA+r_info | ✓ | ✓ | 0.0 | Partner information incentive alone |
| C: MAPPO+BCA+r_info | ✓ | ✓ | 0.05 | Full Dual-Information (partner + opponent penalty) |

**Question chain:**
- A vs SL → value of RL itself (with proper bid understanding)
- B vs A → incremental value of partner information shaping
- C vs A → incremental value of full dual-information
- C vs B → isolated value of β opponent penalty term
- All vs SL → overall improvement landscape

### 2.2 Convention Drift Sweep (Independent Contribution)

Separate experiment, run on **Agent A config only** (no r_info, pure MAPPO+BCA):

| λ_KL | Purpose |
|------|---------|
| 0.0 | Maximum drift (unconstrained RL) |
| 0.1 | Light constraint |
| 0.3 | Moderate constraint |
| 0.5 | Strong constraint |
| 1.0 | Near-SL behavior |

For each λ, measure:
1. **DDS regret** (opponent-independent: how good is the contract?)
2. **vs-SL IMP** (opponent-dependent: includes WBridge5/SL confusion by drifted bids)
3. **Drift advantage** = (2) - (1)

This produces the "convention drift Pareto frontier" figure — the paper's most novel methodological contribution.

### 2.3 Statistical Protocol

- **Seeds:** 5 per configuration (42, 123, 456, 789, 2024)
- **Evaluation deals:** 2000 paired deals per evaluation
- **Test:** Wilcoxon signed-rank (per-deal IMP, non-parametric)
- **Reporting:** mean ± 95% CI (bootstrap)

### 2.4 What to measure for each agent

| Metric | Source | Purpose |
|--------|--------|---------|
| IMP vs DDS optimal | Eval | Primary outcome |
| IMP vs SL baseline | Eval (paired) | Relative improvement |
| Partner info gain | BeliefNetwork diagnostic | Communication quality |
| D_KL(π ∥ π_SL) | KL computation | Protocol compliance |
| Belief accuracy (top-13) | BeliefNetwork validation | Belief quality during RL |
| Entropy | Policy logits | Exploration health |

---

## Part 3: Execution Order

```
Step 1: Modify sl_pretrain_bca.py Stage A (add opponent targets)
        Modify policy_net.py constants (BELIEF_FEAT_DIM=96, BELIEF_OBS_DIM=397)
        Modify Stage B to query both partner and RHO
        Retrain belief net + 397-dim actor
        Output: sl_base_bca_v2.pt
        Time: ~4-6 hours on Colab

Step 2: Modify subgame_validation.py (all agents use BCA)
        Quick sanity run: 1 seed, 3 rounds
        Verify: A trains stably, B trains stably, belief accuracy maintained
        Time: ~2-3 hours

Step 3: Convention drift sweep (Agent A only, 5 λ values × 5 seeds)
        This can run in parallel / overnight
        Time: ~24 hours total

Step 4: A vs B vs C experiment (5 seeds each, 3 configs)
        The core hypothesis test: does r_info help? does β add value?
        Time: ~18 hours total

Step 5: Analysis + figures
        - Convention drift Pareto frontier
        - A vs B IMP comparison with CI
        - Partner info gain comparison
        - Belief accuracy trajectory
```

---

## Part 4: How This Maps to Paper Contributions

### Contribution 1: Convention Drift Quantification
- **Experiment:** Step 3 (λ sweep)
- **Key figure:** Pareto frontier (DDS regret vs vs-SL IMP vs λ)
- **Claim:** First quantification of illegitimate advantage from convention drift in bridge AI
- **Prior work context:** JPS reviewer raised concern (2020), Qiu acknowledged (2024), none quantified

### Contribution 2: Communication-Outcome Disconnect (from prior work)
- **Experiment:** Already done (P97d results)
- **Claim:** r_info improves communication (+10.8% info gain) but not decisions — information-theoretic ≠ decision-theoretic optimality
- **Note:** These results were obtained WITHOUT BCA. They motivated BCA.

### Contribution 3: BCA as Methodological Standard + Dual-Information Fair Test
- **Experiment:** Step 4 (A vs B vs C, all with BCA)
- **If B > A:** Partner information shaping has value when perception-action loop is closed
- **If C > B:** Opponent penalty (β) provides additional value beyond partner-only
- **If C ≈ B > A:** Partner term is the key driver; opponent term is redundant
- **If B ≈ A:** Belief features alone (via BCA) already capture what r_info teaches — the structured representation IS the solution
- **Either way:** BCA is shown to be a necessary baseline for fair bridge AI evaluation

---

## Key Simplifications (What We're NOT Doing)

| Dropped | Reason |
|---------|--------|
| LHO belief features in actor | LHO acts after you; pre-inference has limited value |
| FSP opponents with belief net | Engineering cost too high, marginal research value |
| Stayman multi-seed | Null result confirmed; one paragraph in paper suffices |
| Full-game experiments | Competitive subgame sufficient for thesis; full game = future work |
| Convention-sharing evaluation | Interesting but too ambitious; mention as future work |

---

## Part 6: Conversation Summary (for continuity in new session)

This section documents the full chain of reasoning from the conversation that produced this plan. It covers narrative strategy, prior work analysis, experiment design decisions, and architectural choices.

---

### 6.1 Starting Point: Critique of the Preliminary Report's Generalizability

The conversation began with a critical analysis of the preliminary report (Preliminary_Report_Bridge_COMA_v2.pdf). The core concern: **does this research contribute to MARL, or is it just bridge AI with a MARL veneer?**

**Diagnosis of the current report's weakness:**
- The generalizability argument is "declarative, not structural" — the report mentions air traffic control, finance, medical handoffs as analogous domains, but never formally defines what makes these domains structurally similar
- Evaluation metrics (DDS regret, IMP) are bridge-specific; non-bridge MARL reviewers see them as "some game's score"
- The communication-outcome disconnect is presented as an engineering limitation (actor can't use belief) rather than a scientific finding

**Proposed fix — reposition the narrative:**
- Don't tell the story as "we improved bridge AI using MARL"
- Tell it as "we identified a new MARL problem class (protocol-constrained communication), formalized it, and validated findings on bridge as testbed"
- Bridge is the experimental platform, not the research object
- The research object is protocol-constrained communication as a problem class

**Key narrative strategies identified from literature review:**
1. **Problem-First Framing** (like COMA → StarCraft is testbed, not research object)
2. **Formalize-then-Validate** (like BAD → define public belief MDP first, then test on Hanabi)
3. **Exposing a Blind Spot** (like LangGround → existing emergent communication is uninterpretable)

**Recommended contribution structure (three layers):**
1. Conceptual (domain-agnostic): Protocol-constrained communication as problem class; dual-information principle
2. Method (algorithm-level): r_info + KL constraint + BCA as concrete implementation
3. Empirical (domain-specific): Convention drift quantification, communication-outcome disconnect, belief stabilization

---

### 6.2 Gemini's Analysis and Claude's Response

A document from Gemini was provided with three "升华方向" for the Convention Card concept:
1. Solving ZSC's "secret handshake" problem via forced semantic alignment
2. Human-AI teaming: auditable communication
3. Adversarial information control via prior asymmetry

**Claude's agreements with Gemini:**
- The three directions are all reasonable
- The financial reporting analogy (GAAP/IFRS as convention card) is structurally isomorphic to bridge

**Claude's disagreements with Gemini:**
- "Public Knowledge Constraint" is a worse name than "Protocol-Constrained Communication" (conflicts with Aumann's common knowledge in game theory)
- Gemini conflates Belief Network's role: BN is not the "forced semantic alignment mechanism" — KL constraint is. BN is the detection tool and r_info infrastructure.

**Key question identified that reviewers will ask:**
"How is your KL constraint different from RLHF's KL penalty?"

**Answer:** In RLHF, π_ref is private (only trainer knows) and KL prevents reward hacking. In protocol-constrained communication, π_ref is publicly shared (all parties know), and KL prevents rule violation. Convention drift (~2 IMP advantage) is empirical evidence of this structural difference — in RLHF there is no "opponent who benefits from your policy drift."

---

### 6.3 Prior Work Verification: Convention Drift in Bridge AI Literature

Systematic search of three key bridge AI papers to verify the claim "none discuss or control for this confound":

**JPS (Tian et al., NeurIPS 2020):**
- Authors explicitly acknowledge: "WBridge5 conforms to human convention but JPS can be creative"
- A NeurIPS reviewer directly questioned this: "I have to wonder if this is a misleading comparison, with WBridge5 forced to comply with conventions while JPS is allowed to deviate"
- Author response: repeated the acknowledgment, did no quantification, left as "future work"
- **Conclusion: Awareness exists, no quantification**

**Kita et al. (IEEE CoG 2024):**
- No discussion of convention drift at all
- SL pretrain on SAYC, RL via FSP, evaluate against WBridge5 (which uses a different system)
- **Conclusion: No awareness**

**Qiu et al. (IEEE/CAA JAS 2024):**
- One key sentence: "The conformity of our agents significantly reduces the potential for winning IMPs by confusing WBridge5 with unexpected calls, which is illegal in real-life tournaments"
- No quantification of conformity degree, no KL measurement
- **Conclusion: Partial awareness, no quantification**

**Recommended citation language (more precise than current report):**
"Prior bridge AI work has shown varying degrees of awareness of this issue. Tian et al. (2020) acknowledge that 'WBridge5 conforms to human convention but JPS can be creative,' and a NeurIPS reviewer explicitly questioned whether the comparison was misleading. Qiu et al. (2024) note that winning IMPs by 'confusing WBridge5 with unexpected calls...is illegal in real-life tournaments.' Kita et al. (2024) do not discuss the issue. To our knowledge, no prior work has quantified the magnitude of the illegitimate advantage arising from convention drift."

---

### 6.4 Research Positioning: Honest vs. Overclaimed

**Titus's concern:** "Isn't claiming we 'defined a new MARL problem class' overclaiming? We started from bridge, not from MARL."

**Resolution:** The honest narrative is "from specific to abstract":
- "We started from bridge bidding optimization, encountered unexpected phenomena (convention drift, communication-outcome disconnect), realized these are manifestations of a broader problem class not systematically treated in MARL literature"
- This "discovery narrative" is more credible than "we defined a problem class and validated it"
- Precedent: Foerster discovered public belief's importance on Hanabi, then formalized public belief MDP

**Key test for whether the problem class is "real":**
- Before you proposed it, did others encounter the same difficulty? → Yes (JPS reviewer, Qiu's acknowledgment)
- Can you make testable predictions for other domains? → Yes ("any SL→RL pipeline with shared protocol should exhibit convention drift proportional to KL divergence")

---

### 6.5 Report and Proposal: Deferred Until After BCA Experiments

**Decision:** Do not rewrite the preliminary report or research proposal now. Wait for BCA experiment results, which determine the paper's story:

**Route 1 (BCA enables r_info to work → B or C > A in IMP):**
Three-act story: discover problem (drift) → diagnose (communication-outcome disconnect) → solve (BCA + r_info)

**Route 2 (BCA alone sufficient → B ≈ C ≈ A, all > SL):**
Methodological contribution: convention drift quantification + BCA as standard baseline + negative result on r_info (task reward already implicitly incentivizes informative communication)

**Both routes are publishable.** Route 1 is stronger for NeurIPS/ICML. Route 2 fits AAMAS/IEEE CoG.

**Two deliverables needed after experiments:**
1. Preliminary report for Bristol (school requirement, evaluate research feasibility)
2. PPT for collaborating supervisor (show findings, argue research potential)

---

### 6.6 The BCA Positioning Debate

**Initial framing:** BCA as experimental treatment (comparing BCA vs non-BCA agents)

**Titus's correction:** "BCA is not a tool we use to improve the model — it's a minimum capability. ALL agents must understand bids. Otherwise our research is no different from prior bridge AI work."

**This fundamentally changed the experiment design:**
- ALL agents (including SL baseline and Agent A) use BCA
- The experimental variable is r_info only
- BCA becomes infrastructure, not contribution
- Fairness concern eliminated — everyone has the same understanding capability

**Implication for MARL framing:** The contribution is not "we added belief conditioning" (BAD did this in 2019). The contribution is "we built a properly-equipped experimental platform for testing information-theoretic communication incentives in a protocol-constrained dual-audience setting."

---

### 6.7 The r_info Value Question

**Concern raised:** If BCA gives agents understanding capability, and MAPPO+task reward can learn good policies, is r_info redundant?

**Analysis:**
- r_info teaches "bid informatively for your partner" → but IMP reward already implicitly rewards this (informative bids → better contracts → higher IMP)
- If BCA is good enough, r_info may provide zero incremental value
- This would be a legitimate scientific finding, not a failure

**But this concern is premature:**
- All previous r_info experiments were done WITHOUT BCA (perception-action loop was broken)
- r_info has never been tested under fair conditions
- The experiment must be run before concluding anything

**Resolution:** Keep r_info as the core experimental variable. Run the experiment. Interpret results after.

**MARL value of r_info regardless of outcome:**
- If works: Wiretap-channel-inspired intrinsic reward improves coordination in protocol-constrained settings → new MARL contribution
- If doesn't work: In constrained policy spaces, task reward already implicitly contains information-theoretic incentives → valuable negative result for MARL reward shaping literature

---

### 6.8 The 349→397 Dimension Decision

**Problem identified:** SL baseline only queries belief net for partner (48-dim), not for opponents. This means SL actor learns to imitate human experts (who understand ALL players' bids) using strictly less information. SL quality is artificially capped.

**Chain of reasoning:**
1. "opponent combined belief is mathematically redundant with partner belief" → TRUE for combined opponents
2. BUT P(card in RHO specifically) ≠ 1 - P(card in partner) → RHO-specific inference IS new information
3. Human experts process RHO's bid before deciding → SL should have this capability too
4. Full Disclosure principle requires information AVAILABILITY, not that opponents USE it → FSP opponents using prior is acceptable (they choose not to look at convention card)
5. But SL baseline must have proper understanding → SL needs RHO belief features

**Final decision: 397-dim actor input**
- 301 (base obs) + 48 (partner belief) + 48 (RHO belief) = 397
- BeliefNetwork architecture: no change (already supports arbitrary observer/target)
- SL pretrain Stage A: add opponent target training samples (3× sample size)
- SL pretrain Stage B: query belief net twice per step (partner + RHO)
- All agents share single 397-dim SL checkpoint

**Why RHO not LHO:**
- RHO bid immediately before you → maximum information relevance
- LHO acts after you → pre-inference has limited decision value
- Matches real bridge cognition

**FSP pool:** Opponents use 96-dim prior (two concatenated 48-dim uniform priors). They don't actively interpret our bids. This is a known limitation documented in the plan but acceptable because:
- Full Disclosure = information availability, not mandatory utilization
- FSP weakness is conservative (makes our results harder to achieve, not easier)
- Affects all agents equally → relative comparisons remain fair

---

### 6.9 Final Experiment Matrix

| Agent | BCA (397-dim) | r_info | β | Role |
|-------|---------------|--------|---|------|
| SL baseline | ✓ | ✗ | — | Reference convention (zero-drift anchor) |
| A: MAPPO+BCA | ✓ | ✗ | — | RL improvement with proper understanding |
| B: MAPPO+BCA+r_info | ✓ | ✓ | 0.0 | Partner information incentive alone |
| C: MAPPO+BCA+r_info | ✓ | ✓ | 0.05 | Full Dual-Information |

Plus: Convention drift sweep (Agent A config, λ ∈ {0, 0.1, 0.3, 0.5, 1.0}, 5 seeds each)

**Question chain:**
- A vs SL → value of RL with proper bid understanding (sanity check)
- B vs A → **core RQ: does information-theoretic communication incentive improve coordination?**
- C vs B → **core RQ: does opponent leakage penalty add value?**
- Convention drift sweep → independent methodological contribution

---

### 6.10 Open Issues for Next Session

1. **FSP opponent belief grounding:** Agent C's β term penalizes opponent leakage, but FSP opponents don't exploit leaked info. Is β's gradient signal consistent with task reward? This needs honest discussion in the paper as a limitation.

2. **init_from strategy for 397-dim:** Current sl_pretrain_bca.py supports loading 301-dim weights into 349-dim model with zero-init. Need to extend this for 349→397 or 301→397 if needed.

3. **Belief net training data volume:** Stage A samples triple (partner + LHO + RHO targets). Need to verify belief net quality doesn't degrade with the larger, more diverse training set.

4. **Competitive subgame vs full game:** All experiments currently planned for 1H-1S subgame only. Full game is deferred to future work. Is this sufficient for a publishable paper?

5. **The "BCA is infrastructure, r_info is contribution" framing:** Need to verify this framing holds up. If r_info doesn't work AND BCA alone doesn't beat SL, we have a problem — neither the infrastructure nor the method demonstrates value.
