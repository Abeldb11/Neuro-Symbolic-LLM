"""Run Stage A2: the A1 residual adapter plus the PC-inspired auxiliary losses.

Wiring only: builds the configs, substrate, adapter and the flat A2 parameter tree
(adapter + auxiliary heads), then runs the shared loop in ``stages/stage_base.py``
and checks the base is unchanged. The objective is ``stages/stage_A/a2.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script: put the repo root on sys.path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frozenllm.substrate import FrozenSubstrate  # noqa: E402
from residual import ResidualAdapter  # noqa: E402
from stages.results import RunWriter, default_run_name, run_metadata  # noqa: E402
from stages.stage_A.a2 import (  # noqa: E402
    ARMS,
    A2Config,
    a2_audit,
    a2_objective,
    apply_arm,
    audit_ok,
    audit_to_dict,
    format_audit,
    init_a2_params,
    param_counts,
)
from stages.stage_base import (  # noqa: E402
    Domain,
    RunConfig,
    evaluate,
    evaluate_checkpoint,
    load_configs,
    load_token_blocks,
    make_eval_set,
    run_sequential,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml"

log = logging.getLogger("stage_a2")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--rank", type=int, default=None, help="Residual adapter rank override."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Epochs for every domain (smoke runs)."
    )
    parser.add_argument(
        "--arm",
        choices=ARMS,
        default=None,
        help="Ablation arm: which auxiliary terms are on (default: as in the YAML).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sets adapter.seed, training.seed and objective.seed.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run the A2 checkpoints on the first training batch before training; "
        "stop with exit code 1 if one fails; write audit.json to the run directory.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run the A2 checkpoints, print them as JSON and exit (no training).",
    )
    # Explicit weights override the arm and the YAML.
    for flag, help_text in (
        ("--lambda-kl", "KL weight."),
        ("--lambda-wd", "Adapter weight decay."),
        ("--lambda-cp", "Contrastive-future weight (0 switches the term off)."),
        ("--lambda-stab", "Stability weight (0 switches the term off)."),
    ):
        parser.add_argument(flag, type=float, default=None, help=help_text)
    parser.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="Evaluate a saved checkpoint instead of training.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate when using --eval-checkpoint (default: test).",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Override eval_batches setting.",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    adapter, training, domains = cfg.adapter, cfg.training, cfg.domains
    if args.rank is not None:
        adapter = replace(adapter, rank=args.rank)
    if args.model is not None:
        training = replace(training, model=args.model)
    if args.output_dir is not None:
        training = replace(training, output_dir=args.output_dir)
    if args.epochs is not None and args.steps is not None:
        raise ValueError("Cannot specify both --epochs and --steps. Use --epochs.")
    if args.steps is not None:
        raise ValueError("--steps has been replaced by --epochs. Use --epochs.")
    if args.epochs is not None:
        if args.epochs < 0:
            raise ValueError(f"epochs must be >= 0, got {args.epochs}")
        training = replace(training, epochs=args.epochs)
        domains = tuple(replace(d, epochs=args.epochs) for d in domains)
    objective = cfg.objective
    if args.arm is not None:
        objective = apply_arm(objective, args.arm)
    if args.seed is not None:
        adapter = replace(adapter, seed=args.seed)
        training = replace(training, seed=args.seed)
        objective = replace(objective, seed=args.seed)
    updates = {
        name: value
        for name, value in (
            ("lambda_kl", args.lambda_kl),
            ("lambda_wd", args.lambda_wd),
        )
        if value is not None
    }
    if updates:
        objective = replace(objective, **updates)
    if args.lambda_cp is not None:
        objective = replace(
            objective, cpc=replace(objective.cpc, weight=args.lambda_cp)
        )
    if args.lambda_stab is not None:
        objective = replace(
            objective, stab=replace(objective.stab, weight=args.lambda_stab)
        )
    return replace(
        cfg, adapter=adapter, objective=objective, training=training, domains=domains
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)
    cfg = apply_overrides(load_configs(args.config, A2Config), args)
    train_cfg = cfg.training
    log.info("adapter=%s", cfg.adapter)
    log.info("objective=%s", cfg.objective)
    log.info("training=%s", train_cfg)
    log.info("domains=%s", [d.name for d in cfg.domains])

    substrate = FrozenSubstrate(train_cfg.model)
    arch = substrate.architecture
    max_pos = arch.max_position_embeddings
    if max_pos is not None and train_cfg.seq_len > max_pos:
        raise ValueError(f"seq_len={train_cfg.seq_len} exceeds {max_pos} positions.")

    adapter = ResidualAdapter(cfg.adapter, arch)
    params = init_a2_params(cfg.objective, adapter)
    counts = param_counts(params)
    n_trainable = counts["total"]
    n_frozen = sum(p.numel() for p in substrate.params.values())
    log.info(
        "model=%s L=%d d=%d layers=%s rank=%d trainable=%d (%s) frozen=%d ratio=%.3e",
        train_cfg.model,
        arch.num_layers,
        arch.hidden_size,
        list(adapter.layers),
        cfg.adapter.rank,
        n_trainable,
        counts,
        n_frozen,
        n_trainable / n_frozen,
    )

    frozen = substrate.verify_frozen()
    if not frozen["params_unchanged"]:
        log.error("Substrate not frozen before training: %s", frozen)
        return 1

    if args.eval_checkpoint is not None:
        log.info(
            "Evaluating checkpoint %s on split=%s",
            args.eval_checkpoint,
            args.split,
        )
        eval_batches = args.eval_batches or train_cfg.eval_batches
        results = evaluate_checkpoint(
            substrate,
            adapter,
            args.eval_checkpoint,
            cfg.domains,
            split=args.split,
            seq_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size,
            eval_batches=eval_batches,
        )
        print(json.dumps(results, indent=2))
        return 0

    # Load data before creating the run directory, so bad datasets fail fast.
    domains = []
    for d in cfg.domains:
        train_blocks = load_token_blocks(substrate, d, d.train_split, train_cfg.seq_len)
        eval_blocks = load_token_blocks(substrate, d, d.eval_split, train_cfg.seq_len)
        domains.append(
            Domain(
                name=d.name,
                train_blocks=train_blocks,
                eval_set=make_eval_set(
                    eval_blocks, train_cfg.batch_size, train_cfg.eval_batches
                ),
                epochs=train_cfg.epochs if d.epochs is None else d.epochs,
            )
        )
        log.info(
            "domain=%s train_blocks=%d eval_batches=%d",
            d.name,
            len(train_blocks),
            len(domains[-1].eval_set),
        )

    audit = None
    if args.audit or args.audit_only:
        first = domains[0].train_blocks[: train_cfg.batch_size]
        audit = a2_audit(substrate, adapter, cfg.objective, jnp.asarray(first))
        log.info("A2 audit on %d sequences:\n%s", len(first), format_audit(audit))
        if args.audit_only:
            print(json.dumps(audit_to_dict(audit), indent=2))
            return 0 if audit_ok(audit) else 1
        if not audit_ok(audit):
            log.error("A2 audit failed; not training.")
            return 1

    prefix = "a2" if args.arm is None else f"a2-{args.arm}"
    if args.seed is not None:
        prefix += f"-s{args.seed}"
    run_name = args.run_name or default_run_name(prefix)
    with RunWriter(train_cfg.output_dir, run_name) as writer:
        log.info("run_dir=%s", writer.dir)
        writer.write_config(cfg.to_dict())
        if audit is not None:
            writer.write_json("audit.json", audit_to_dict(audit))
        writer.write_json(
            "meta.json",
            {
                **run_metadata(),
                "stage": "A2",
                "arm": args.arm,
                "seed": args.seed,
                "model": train_cfg.model,
                "num_layers": arch.num_layers,
                "hidden_size": arch.hidden_size,
                "adapted_layers": list(adapter.layers),
                "trainable_params": n_trainable,
                "trainable_params_by_group": counts,
                "frozen_params": n_frozen,
                "epochs": {d.name: d.epochs for d in domains},
            },
        )
        params, summary = run_sequential(
            substrate,
            adapter,
            params,
            domains,
            a2_objective,
            cfg.objective,
            train_cfg,
            writer,
        )
        frozen = substrate.verify_frozen()
        summary["params_unchanged"] = frozen["params_unchanged"]

        # Final test-set evaluation across domains for paper-table results
        test_eval_batches = args.eval_batches or train_cfg.eval_batches
        test_results = {}
        for d in cfg.domains:
            test_blocks = load_token_blocks(
                substrate, d, d.test_split, train_cfg.seq_len
            )
            test_set = make_eval_set(
                test_blocks, train_cfg.batch_size, test_eval_batches
            )
            test_results[d.name] = evaluate(substrate, adapter, params, test_set)
            writer.log("test", domain=d.name, **test_results[d.name])
        summary["test"] = test_results
        writer.write_json("results.json", summary)

    log.info("mean_seen_loss=%s", summary["mean_seen_loss"])
    log.info("mean_forgetting=%s", summary["mean_forgetting"])
    log.info("test_results=%s", summary["test"])
    log.info("verify_frozen=%s", frozen)
    if not frozen["params_unchanged"]:
        log.error("Base parameters changed during training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())