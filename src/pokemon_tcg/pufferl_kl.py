"""PuffeRL + KL-to-prior anchor (side-venv only) — rl_pipeline_v1.md §3.2.

`PuffeRLPriorKL.train()` is a VERBATIM COPY of pufferlib 3.0's
`pufferl.PuffeRL.train` (installed at .venv-ppo/.../pufferlib/pufferl.py,
lines 312-451) with additions fenced by `>>> KL ANCHOR` markers:
a penalty `kl_coef * KL(pi_current || pi_prior)` added to the loss (plus a
per-update `self.losses` export so the driver can log the KL every update),
where the prior starts as the frozen BC/REWEIGHT checkpoint from before any
PPO. Their clip constrains each update step; this constrains total drift
across the run — the term stock pufferl does not have (its approx_kl is
logged only).

The prior is not necessarily static: `retarget_prior()` implements the
checkpoint-promotion ratchet (Orbit Wars 1st place / Lux AI S1) — when the
live policy decisively beats the reference in mirrored eval matches
(pokemon_tcg.promotion.evaluate_gate), the reference becomes a frozen copy
of the live policy and the KL pull continues against the NEW anchor. The
pull itself is continual: it stays on for the entire run, before and after
every promotion.

If pufferlib is upgraded, re-diff this copy against upstream train().

Both current and prior policies mask illegal options to a FINITE -1e9
(pokemon_tcg.puffer_policy.NEG), so softmax/log_softmax here are NaN-free by
construction and no torch.where gating is needed (unlike the -inf lesson in
pokemon_tcg/ppo.py).
"""

from __future__ import annotations

import time
from collections import defaultdict

import torch

import pufferlib.pytorch
from pufferlib.pufferl import PuffeRL, compute_puff_advantage

# Anchor math lives in the pufferlib-free kl_math module so the main venv's
# test suite (tests/test_kl_mask.py) exercises the exact functions this
# trainer ships with — see that module's docstring.
from .kl_math import masked_kl, masked_kl_per_row, masks_agree  # noqa: F401
from .puffer_env import OBS_LAYOUT, _OFFSETS

# select_context's offset in the packed flat obs — used to attribute per-row
# KL/entropy to the SelectContext it happened in (rl_pipeline_v2 §3.2: the
# aggregate hides single-context entropy collapse; the per-context lines are
# the stop-signal instrument).
_CTX_OFF = int(_OFFSETS[[i for i, (k, _, _) in enumerate(OBS_LAYOUT)
                         if k == "select_context"][0]])
_N_CTX = 96  # SELECT_CONTEXT_VOCAB_SIZE headroom


class PuffeRLPriorKL(PuffeRL):
    def __init__(self, config, vecenv, policy, logger=None, *,
                 kl_prior_policy=None, kl_coef: float = 0.05) -> None:
        super().__init__(config, vecenv, policy, logger)
        assert kl_prior_policy is not None, "pass the frozen prior policy"
        self.kl_prior_policy = kl_prior_policy.eval()
        for p in self.kl_prior_policy.parameters():
            p.requires_grad_(False)
        self.kl_coef = kl_coef
        self._mask_checked = False

    @torch.no_grad()
    def retarget_prior(self, live_policy) -> None:
        """Promotion (the ratchet): pi_ref <- frozen copy of the live policy.

        ``load_state_dict`` copies VALUES into the existing frozen
        Parameters -- requires_grad stays False, eval mode stays, and the
        optimizer never held these tensors. The KL pull continues unchanged,
        now anchored to the new reference.
        """
        self.kl_prior_policy.load_state_dict(live_policy.state_dict())
        self._mask_checked = False  # re-verify mask agreement on the next minibatch

    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        losses = defaultdict(float)
        config = self.config
        device = config['device']

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1

        # >>> KL ANCHOR: per-SelectContext KL/entropy accumulators (§3.2) >>>
        device_t = torch.device(device)
        ctx_kl_sum = torch.zeros(_N_CTX, device=device_t)
        ctx_ent_sum = torch.zeros(_N_CTX, device=device_t)
        ctx_n = torch.zeros(_N_CTX, device=device_t)
        # <<< KL ANCHOR <<<

        for mb in range(self.total_minibatches):
            profile('train_misc', epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(self.values, self.rewards,
                self.terminals, self.ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            profile('train_copy', epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile('train_forward', epoch)
            if not config['use_rnn']:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            logits, newvalue = self.policy(mb_obs, state)
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

            profile('train_misc', epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            adv = advantages[idx]
            adv = compute_puff_advantage(mb_values, mb_rewards, mb_terminals,
                ratio, adv, config['gamma'], config['gae_lambda'],
                config['vtrace_rho_clip'], config['vtrace_c_clip'])
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss

            # >>> KL ANCHOR (fenced change vs upstream train()) >>>
            with torch.no_grad():
                prior_logits, _ = self.kl_prior_policy(mb_obs, state)
            if not self._mask_checked:
                # One-time (per prior) guard: both distributions must be
                # softmaxed over the IDENTICAL legal-action mask. Structurally
                # guaranteed today (both forwards mask from the same mb_obs's
                # opt_mask), but a future prior that masks differently -- or
                # not at all -- would silently corrupt the anchor gradient
                # with probability mass on illegal actions. Unit-tested in
                # tests/test_kl_mask.py against the same kl_math functions.
                assert masks_agree(logits.detach(), prior_logits), (
                    "KL anchor mask mismatch: current and prior policies "
                    "disagree on which options are illegal for the same obs")
                self._mask_checked = True
            kl_vec = masked_kl_per_row(logits, prior_logits)
            kl_prior = kl_vec.mean()
            loss = loss + self.kl_coef * kl_prior
            losses['kl_to_prior'] += kl_prior.item() / self.total_minibatches
            # Per-SelectContext attribution (detached; §3.2 stop-signal
            # instrument). mb_obs is already flat [N, OBS_SIZE] here when
            # use_rnn is False (reshaped above before the forward).
            with torch.no_grad():
                flat_obs = mb_obs.reshape(-1, mb_obs.shape[-1])
                ctx = flat_obs[:, _CTX_OFF].long().clamp_(0, _N_CTX - 1)
                ctx_kl_sum += torch.bincount(
                    ctx, weights=kl_vec.detach().reshape(-1), minlength=_N_CTX)
                ctx_ent_sum += torch.bincount(
                    ctx, weights=entropy.detach().reshape(-1), minlength=_N_CTX)
                ctx_n += torch.bincount(ctx, minlength=_N_CTX).float()
            # <<< KL ANCHOR <<<

            self.amp_context.__enter__() # TODO: AMP needs some debugging

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()

            # Logging
            profile('train_misc', epoch)
            losses['policy_loss'] += pg_loss.item() / self.total_minibatches
            losses['value_loss'] += v_loss.item() / self.total_minibatches
            losses['entropy'] += entropy_loss.item() / self.total_minibatches
            losses['old_approx_kl'] += old_approx_kl.item() / self.total_minibatches
            losses['approx_kl'] += approx_kl.item() / self.total_minibatches
            losses['clipfrac'] += clipfrac.item() / self.total_minibatches
            losses['importance'] += ratio.mean().item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile('learn', epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Reprioritize experience
        profile('train_misc', epoch)
        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        losses['explained_variance'] = explained_var.item()

        # >>> KL ANCHOR (fenced change vs upstream train()) >>>
        # Upstream only publishes self.losses on the rate-limited dashboard
        # path; publish every update so the driver can log kl_to_prior per
        # update (same values the dashboard would show, just not throttled).
        self.losses = losses
        # Per-SelectContext means for this update, for the driver's JSONL log
        # (contexts actually seen only).
        nz = torch.nonzero(ctx_n > 0).flatten()
        self.per_context = {
            int(c): {
                "kl": round(float(ctx_kl_sum[c] / ctx_n[c]), 5),
                "entropy": round(float(ctx_ent_sum[c] / ctx_n[c]), 5),
                "n": int(ctx_n[c]),
            }
            for c in nz.tolist()
        }
        # <<< KL ANCHOR <<<

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        return logs
