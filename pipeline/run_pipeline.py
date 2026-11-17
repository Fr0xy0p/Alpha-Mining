"""
run_pipeline.py — AlphaMiningV2 unified closed-loop factor mining pipeline

Merges the four steps into a single automated entry point:

  Phase 1  PROBE-GEN    Generate probe batches (each Core uses representative_values)
  Phase 2  PROBE-RUN    Execute backtests on probe batches
  Phase 3  SCHEDULE     Aggregate results by Core; decide EXPAND / WATCH / ABANDON
  Phase 4  EXPAND-RUN   Generate full parameter batches for EXPAND Cores and run backtests

Usage Examples:

  # Simplest run (all defaults):
  python run_pipeline.py --dataset-id option8

  # Specific templates and decision thresholds:
  python run_pipeline.py --dataset-id option8 --template-ids TPL_GROUP_IVHV_SMOOTH_V1 \\
    --expand-min-sharpe 1.0 \\
    --expand-max-turnover 0.7

  # Re-run only scheduling and expansion (skip completed probe generation and backtest):
  python run_pipeline.py --dataset-id option8 --skip-probe-gen --skip-probe-run

  # Dry-run mode: print scheduling decisions without generating expand batches:
  python run_pipeline.py --dataset-id option8 --dry-run

  # Probe-only mode: run Phase 1 & 2 without scheduling or expansion:
  python run_pipeline.py --dataset-id option8 --probe-only
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path


# ─── Add project root to sys.path (supports running from any working dir) ─────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ─── Lazy import modules (after parse_args to prevent import errors masking --help)
# Actual imports done inside phase functions and main().


# ═══════════════════════════════════════════════════════════════════════════════
# Logging Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_pipeline_logger(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("run_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _section(logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Probe Batch Generation
# ═══════════════════════════════════════════════════════════════════════════════

def phase_probe_gen(args, logger: logging.Logger) -> list[Path]:
    """Call core logic of main.py with probe_mode=True to generate probe batches.

    Directly imports and invokes functions to avoid subprocess overhead.
    Returns list of generated batch file paths.
    """
    from pipeline.main import (
        load_template_catalog,
        validate_template_names,
        load_operator_names,
        load_common_operator_slot_mappings,
        parse_selected_template_ids,
        load_slot_overrides,
        load_settings_grid,
        iter_settings,
        iter_alpha_requests,
        validate_operator_slots,
        write_factor_batches,
        SIMULATION_SETTINGS,
        DEFAULT_TEMPLATE_DOC,
        DEFAULT_OPERATORS_DOC,
        DEFAULT_OPERATOR_SLOT_MAP_DOC,
    )
    from pipeline.datafields_store import fetch_and_store_datafields

    _section(logger, "PHASE 1 — PROBE GENERATION")
    probe_output_dir = Path(args.probe_batches_dir)
    probe_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching dataset fields for dataset_id=%s ...", args.dataset_id)

    # Fetch dataset fields from WorldQuant Brain API
    dataset_fields_df = fetch_and_store_datafields(
        dataset_id=args.dataset_id,
        data_type=args.data_type,
        output_dir=Path(args.datafields_dir),
        instrument_type=args.instrument_type,
        region=args.region,
        delay=args.delay,
        universe=args.universe,
        search=args.search,
    )
    if dataset_fields_df.empty or "id" not in dataset_fields_df.columns:
        raise RuntimeError("No data fields returned from API. Check credentials and dataset_id.")
    dataset_field_ids = dataset_fields_df["id"].dropna().astype(str).tolist()
    logger.info("Fetched %d dataset fields.", len(dataset_field_ids))

    catalog = load_template_catalog(Path(args.template_doc))
    validate_template_names(catalog)
    operator_names = load_operator_names(Path(args.operators_doc))
    common_op_mappings = load_common_operator_slot_mappings(Path(args.operator_slot_map_doc))

    all_template_ids = [t["template_id"] for t in catalog["templates"]]
    selected_ids = parse_selected_template_ids(args.template_ids, all_template_ids)
    slot_overrides = load_slot_overrides(args.slot_overrides_file)
    settings_grid = load_settings_grid(args.settings_grid_file)
    settings_list = list(iter_settings(SIMULATION_SETTINGS, settings_grid))

    # Filter templates applicable to current dataset
    applicable_templates = []
    for t in catalog["templates"]:
        if t["template_id"] not in selected_ids:
            continue
        applicable_datasets = t.get("applicable_datasets", [])
        if isinstance(applicable_datasets, list) and applicable_datasets and args.dataset_id not in applicable_datasets:
            logger.info("Skip template %s (not applicable to dataset %s)", t["template_id"], args.dataset_id)
            continue
        applicable_templates.append(t)

    if not applicable_templates:
        raise RuntimeError("No applicable templates found after dataset filtering.")

    logger.info(
        "Selected %d template(s), %d settings combination(s). Probe mode: ON",
        len(applicable_templates), len(settings_list),
    )

    # Validate operator slots
    for t in applicable_templates:
        tpl_override = slot_overrides.get(t["template_id"], {})
        global_override = slot_overrides.get("global", {})
        merged = {**({} if not isinstance(global_override, dict) else global_override),
                  **({} if not isinstance(tpl_override, dict) else tpl_override)}
        validate_operator_slots(
            template=t,
            operator_names=operator_names,
            overrides=merged,
            common_operator_slot_mappings=common_op_mappings,
        )

    alpha_iter = iter_alpha_requests(
        templates=applicable_templates,
        datafields=dataset_field_ids,
        slot_overrides=slot_overrides,
        common_operator_slot_mappings=common_op_mappings,
        settings_list=settings_list,
        field_role_mode=args.field_role_mode,
        max_per_template=max(1, args.max_per_template),
        max_generated=max(1, args.max_generated),
        probe_mode=True,   # Probe mode enabled
    )

    written_files = write_factor_batches(
        alpha_iter=alpha_iter,
        output_dir=probe_output_dir,
        dataset_id=args.dataset_id,
        template_id=args.template_ids if args.template_ids else "all",
        batch_size=max(1, args.batch_size),
        generation_context={
            "template_ids": [t["template_id"] for t in applicable_templates],
            "field_role_mode": args.field_role_mode,
            "settings_grid": settings_grid,
            "settings_count": len(settings_list),
            "probe_mode": True,
            "pipeline_phase": "probe",
        },
    )

    logger.info("Probe generation complete. Created %d batch file(s) in %s", len(written_files), probe_output_dir)
    return written_files


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 / 4: Execute Backtest (Shared)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_run_backtest(
    input_dir: Path,
    output_dir: Path,
    args,
    logger: logging.Logger,
    phase_label: str = "BACKTEST",
) -> None:
    """Call core backtest_runner functions to execute backtests on all batches in input_dir."""
    from pipeline.backtest_runner import (
        setup_logger as br_setup_logger,
        load_credentials,
        BrainSessionManager,
        iter_unprocessed_batches,
        process_batch_file,
        DEFAULT_CHECKPOINT_DIRNAME,
    )

    _section(logger, f"PHASE — {phase_label}")

    if not input_dir.exists() or not any(input_dir.rglob("*.json")):
        logger.warning("No batch files found in %s. Skipping backtest phase.", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Re-use backtest_runner logger, writing to output_dir/logs
    br_logger = br_setup_logger(output_dir=output_dir, log_level=args.log_level)

    username, password = load_credentials()
    session_manager = BrainSessionManager(
        username=username,
        password=password,
        logger=br_logger,
        relogin_interval_seconds=args.relogin_interval_seconds,
    )
    session_manager.get_session(force_relogin=True)

    max_workers = min(3, max(1, args.max_workers))
    max_retries = max(1, args.max_retries)
    retry_sleep = max(0.0, args.retry_sleep)

    pending = list(iter_unprocessed_batches(input_dir=input_dir, output_dir=output_dir))
    logger.info("Found %d batch file(s) to process.", len(pending))

    failed_batches = []
    for input_file, output_file, checkpoint_file in pending:
        try:
            process_batch_file(
                session_manager=session_manager,
                input_file=input_file,
                output_file=output_file,
                checkpoint_file=output_dir / DEFAULT_CHECKPOINT_DIRNAME / input_file.relative_to(input_dir),
                max_workers=max_workers,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep,
                logger=br_logger,
            )
        except Exception as exc:
            logger.error("Batch processing failed for %s: %s", input_file, exc, exc_info=True)
            failed_batches.append((str(input_file), str(exc)))
            # Continue processing other batches instead of crashing
            continue

    if failed_batches:
        logger.warning("Failed to process %d batch file(s):", len(failed_batches))
        for batch_path, error_msg in failed_batches:
            logger.warning("  - %s: %s", batch_path, error_msg)

    logger.info("%s complete. (processed=%d failed=%d)", phase_label, len(pending) - len(failed_batches), len(failed_batches))


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Adaptive Scheduling (Probe-Expand Decision)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_schedule(args, probe_results_dir: Path, expand_batches_dir: Path, logger: logging.Logger) -> dict:
    """Aggregate probe backtest results, make EXPAND/WATCH/ABANDON decisions,

    and generate full batches for EXPAND Cores.
    Returns the scheduling report dict (and writes to JSON).
    """
    from pipeline.adaptive_scheduler import (
        load_probe_results,
        aggregate_by_core,
        classify_core,
        generate_expand_batch,
        DECISION_EXPAND,
        DECISION_WATCH,
        DECISION_ABANDON,
    )

    _section(logger, "PHASE 3 — ADAPTIVE SCHEDULING")

    if not probe_results_dir.exists():
        raise FileNotFoundError(f"Probe results directory not found: {probe_results_dir}")

    logger.info("Loading probe results from: %s", probe_results_dir)
    all_results = load_probe_results(probe_results_dir)
    logger.info("Loaded %d factor results.", len(all_results))

    if not all_results:
        logger.warning("No probe results found. Skipping scheduling.")
        return {}

    core_stats = aggregate_by_core(all_results)
    logger.info("Found %d unique cores.", len(core_stats))

    decisions: dict[str, list] = {DECISION_EXPAND: [], DECISION_WATCH: [], DECISION_ABANDON: []}

    for core_id, stats in sorted(core_stats.items(), key=lambda x: -x[1]["sharpe_mean"]):
        decision = classify_core(
            core_stats=stats,
            expand_min_sharpe=args.expand_min_sharpe,
            expand_min_fitness=args.expand_min_fitness,
            expand_max_turnover=args.expand_max_turnover,
            watch_min_sharpe=args.watch_min_sharpe,
            min_probe_count=args.min_probe_count,
        )
        stats["decision"] = decision
        decisions[decision].append(stats)

    logger.info(
        "Scheduling decisions: EXPAND=%d  WATCH=%d  ABANDON=%d",
        len(decisions[DECISION_EXPAND]),
        len(decisions[DECISION_WATCH]),
        len(decisions[DECISION_ABANDON]),
    )

    if decisions[DECISION_EXPAND]:
        logger.info("── EXPAND cores ──")
        for s in decisions[DECISION_EXPAND]:
            core_id_str = s.get("core_id") or s.get("pipeline_core_id", "")
            logger.info(
                "  %-60s  sharpe_mean=%.3f  probes=%d",
                core_id_str[:60], s["sharpe_mean"], s["probe_count"],
            )

    if decisions[DECISION_WATCH]:
        logger.info("── WATCH cores ──")
        for s in decisions[DECISION_WATCH]:
            core_id_str = s.get("core_id") or s.get("pipeline_core_id", "")
            logger.info(
                "  %-60s  sharpe_mean=%.3f  probes=%d",
                core_id_str[:60], s["sharpe_mean"], s["probe_count"],
            )

    # Generate expand batches
    expand_success = 0
    expand_fail = 0

    if decisions[DECISION_EXPAND] and not args.dry_run:
        logger.info("── Generating expand batches ──")
        for stats in decisions[DECISION_EXPAND]:
            success = generate_expand_batch(
                pipeline_core_id=stats["pipeline_core_id"],
                pipeline_template_id=stats["pipeline_template_id"],
                dataset_id=args.dataset_id,
                data_type=args.data_type,
                template_doc=args.template_doc,
                output_dir=str(expand_batches_dir),
                slot_overrides_file=args.slot_overrides_file,
                settings_grid_file=args.settings_grid_file,
                dry_run=False,
            )
            if success:
                expand_success += 1
                logger.info("  [OK] Expand batch generated for core: %s", stats["pipeline_core_id"][:60])
            else:
                expand_fail += 1
                logger.error("  [FAIL] Expand batch generation failed for core: %s", stats["pipeline_core_id"][:60])
    elif args.dry_run and decisions[DECISION_EXPAND]:
        logger.info("[DRY-RUN] Would generate %d expand batch(es). Skipping.", len(decisions[DECISION_EXPAND]))

    # Write scheduling report
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "probe_results_dir": str(probe_results_dir),
        "expand_batches_dir": str(expand_batches_dir),
        "thresholds": {
            "expand_min_sharpe": args.expand_min_sharpe,
            "expand_min_fitness": args.expand_min_fitness,
            "expand_max_turnover": args.expand_max_turnover,
            "watch_min_sharpe": args.watch_min_sharpe,
            "min_probe_count": args.min_probe_count,
        },
        "summary": {
            "total_cores": len(core_stats),
            "expand": len(decisions[DECISION_EXPAND]),
            "watch": len(decisions[DECISION_WATCH]),
            "abandon": len(decisions[DECISION_ABANDON]),
            "expand_batches_generated": expand_success,
            "expand_batches_failed": expand_fail,
        },
        "decisions": {
            DECISION_EXPAND: [{k: v for k, v in s.items() if k != "all_metrics"} for s in decisions[DECISION_EXPAND]],
            DECISION_WATCH:  [{k: v for k, v in s.items() if k != "all_metrics"} for s in decisions[DECISION_WATCH]],
            DECISION_ABANDON:[{k: v for k, v in s.items() if k != "all_metrics"} for s in decisions[DECISION_ABANDON]],
        },
    }

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Scheduling report written to: %s", report_path)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args, logger: logging.Logger) -> None:
    start_time = time.monotonic()

    probe_batches_dir  = Path(args.probe_batches_dir)
    probe_results_dir  = Path(args.probe_results_dir)
    expand_batches_dir = Path(args.expand_batches_dir)
    expand_results_dir = Path(args.expand_results_dir)

    logger.info("Pipeline started.")
    logger.info("  dataset_id      : %s", args.dataset_id)
    logger.info("  template_ids    : %s", args.template_ids)
    logger.info("  probe_batches   : %s", probe_batches_dir)
    logger.info("  probe_results   : %s", probe_results_dir)
    logger.info("  expand_batches  : %s", expand_batches_dir)
    logger.info("  expand_results  : %s", expand_results_dir)
    logger.info("  dry_run         : %s", args.dry_run)
    logger.info("  probe_only      : %s", args.probe_only)

    # ── Phase 1: Probe Generation ─────────────────────────────────────────────
    if args.skip_probe_gen:
        logger.info("[SKIP] Phase 1 (probe generation) skipped by --skip-probe-gen.")
    else:
        phase_probe_gen(args, logger)

    # ── Phase 2: Probe Backtest ───────────────────────────────────────────────
    if args.skip_probe_run:
        logger.info("[SKIP] Phase 2 (probe backtest) skipped by --skip-probe-run.")
    else:
        phase_run_backtest(
            input_dir=probe_batches_dir,
            output_dir=probe_results_dir,
            args=args,
            logger=logger,
            phase_label="PROBE BACKTEST (Phase 2)",
        )

    if args.probe_only:
        elapsed = time.monotonic() - start_time
        logger.info("Pipeline stopped after probe phases (--probe-only). Elapsed: %.1fs", elapsed)
        return

    # ── Phase 3: Adaptive Scheduling ──────────────────────────────────────────
    report = phase_schedule(
        args=args,
        probe_results_dir=probe_results_dir,
        expand_batches_dir=expand_batches_dir,
        logger=logger,
    )

    if args.dry_run:
        elapsed = time.monotonic() - start_time
        logger.info("[DRY-RUN] Pipeline finished without expand backtest. Elapsed: %.1fs", elapsed)
        return

    expand_count = report.get("summary", {}).get("expand_batches_generated", 0)
    if expand_count == 0:
        logger.info("No expand batches were generated (0 EXPAND cores). Pipeline complete.")
        elapsed = time.monotonic() - start_time
        logger.info("Total elapsed: %.1fs", elapsed)
        return

    # ── Phase 4: Expand Backtest ──────────────────────────────────────────────
    phase_run_backtest(
        input_dir=expand_batches_dir,
        output_dir=expand_results_dir,
        args=args,
        logger=logger,
        phase_label="EXPAND BACKTEST (Phase 4)",
    )

    elapsed = time.monotonic() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Total elapsed: %.1fs", elapsed)
    logger.info("  Probe results : %s", probe_results_dir)
    logger.info("  Expand results: %s", expand_results_dir)
    logger.info("  Report        : %s", args.report_file)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step: review results with")
    logger.info("  python result_filter.py --results-dir %s --group-by-core", expand_results_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AlphaMiningV2 unified pipeline: probe generation → probe backtest → "
            "adaptive scheduling → expand backtest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Individual phases can also be run standalone:\n"
            "  python main.py --probe          (Phase 1)\n"
            "  python backtest_runner.py       (Phase 2 / 4)\n"
            "  python adaptive_scheduler.py    (Phase 3)\n"
            "  python result_filter.py         (Post-analysis)\n"
        ),
    )

    # ── Dataset & Template ──────────────────────────────────────────────────
    grp_data = parser.add_argument_group("Dataset & Template")
    grp_data.add_argument("--dataset-id", default="pv13",
                          help="Dataset id (e.g. pv13, option8).")
    grp_data.add_argument("--data-type", default="GROUP",
                          help="Data field type filter.")
    grp_data.add_argument("--template-ids", default="ALL",
                          help="Comma-separated template ids, or ALL.")
    grp_data.add_argument("--template-doc", default="template_catalog.json",
                          help="Template catalog JSON file.")
    grp_data.add_argument("--operators-doc", default="wq_operators_cleaned.json",
                          help="WorldQuant operators JSON file.")
    grp_data.add_argument("--operator-slot-map-doc", default="common_operator_slot_mappings.json",
                          help="Common operator slot mappings JSON file.")
    grp_data.add_argument("--slot-overrides-file", default="",
                          help="Optional slot overrides JSON file.")
    grp_data.add_argument("--settings-grid-file", default="",
                          help="Optional simulation settings grid JSON file.")
    grp_data.add_argument("--field-role-mode", choices=["auto", "shared", "distinguish"], default="auto",
                          help="Multi-field role assignment mode.")
    grp_data.add_argument("--instrument-type", default="EQUITY")
    grp_data.add_argument("--region", default="USA")
    grp_data.add_argument("--delay", type=int, default=1)
    grp_data.add_argument("--universe", default="TOP3000")
    grp_data.add_argument("--search", default="",
                          help="Optional search term for field fetch.")

    # ── Generation Scale ────────────────────────────────────────────────────
    grp_gen = parser.add_argument_group("Generation Scale")
    grp_gen.add_argument("--batch-size", type=int, default=500,
                         help="Factors per batch file.")
    grp_gen.add_argument("--max-per-template", type=int, default=5000,
                         help="Generation cap per template.")
    grp_gen.add_argument("--max-generated", type=int, default=20000,
                         help="Global generation cap.")

    # ── Directories ─────────────────────────────────────────────────────────
    grp_dirs = parser.add_argument_group("Directories")
    grp_dirs.add_argument("--datafields-dir", default="datafields_cache",
                          help="Data field cache directory.")
    grp_dirs.add_argument("--probe-batches-dir", default="factor_batches/probe",
                          help="Output directory for probe batch files.")
    grp_dirs.add_argument("--probe-results-dir", default="backtest_results/probe",
                          help="Output directory for probe backtest results.")
    grp_dirs.add_argument("--expand-batches-dir", default="factor_batches/expand",
                          help="Output directory for expand batch files.")
    grp_dirs.add_argument("--expand-results-dir", default="backtest_results/expand",
                          help="Output directory for expand backtest results.")

    # ── Backtest Parameters ─────────────────────────────────────────────────
    grp_bt = parser.add_argument_group("Backtest")
    grp_bt.add_argument("--max-workers", type=int, default=3,
                        help="Parallel backtest workers (max 3, Brain API limit).")
    grp_bt.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts per factor.")
    grp_bt.add_argument("--retry-sleep", type=float, default=2.0,
                        help="Sleep seconds between retries.")
    grp_bt.add_argument("--relogin-interval-seconds", type=int, default=13800,
                        help="Re-login interval in seconds (~3h50m).")

    # ── Scheduling Thresholds ───────────────────────────────────────────────
    grp_sched = parser.add_argument_group("Scheduling Thresholds")
    grp_sched.add_argument("--expand-min-sharpe", type=float, default=1.0,
                           help="Min mean Sharpe to trigger EXPAND.")
    grp_sched.add_argument("--expand-min-fitness", type=float, default=0.45,
                           help="Min mean Fitness to trigger EXPAND.")
    grp_sched.add_argument("--expand-max-turnover", type=float, default=0.7,
                           help="Max mean Turnover to allow EXPAND.")
    grp_sched.add_argument("--watch-min-sharpe", type=float, default=0.5,
                           help="Min mean Sharpe to classify as WATCH (vs ABANDON).")
    grp_sched.add_argument("--min-probe-count", type=int, default=1,
                           help="Min successful probe results required before deciding.")

    # ── Report ──────────────────────────────────────────────────────────────
    grp_report = parser.add_argument_group("Report")
    grp_report.add_argument("--report-file", default="pipeline_report.json",
                            help="Path to write the pipeline scheduling report.")
    grp_report.add_argument("--log-level", default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ── Flow Control ────────────────────────────────────────────────────────
    grp_ctrl = parser.add_argument_group("Pipeline Control")
    grp_ctrl.add_argument("--skip-probe-gen", action="store_true", default=False,
                          help="Skip Phase 1 (probe batch generation). Useful when probe batches already exist.")
    grp_ctrl.add_argument("--skip-probe-run", action="store_true", default=False,
                          help="Skip Phase 2 (probe backtest). Useful when probe results already exist.")
    grp_ctrl.add_argument("--probe-only", action="store_true", default=False,
                          help="Run only Phase 1 and Phase 2, then stop.")
    grp_ctrl.add_argument("--dry-run", action="store_true", default=False,
                          help="Run scheduling analysis but do not generate expand batches or run expand backtest.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Log setup (expand_results_dir as log root to ensure centralized logging)
    log_root = Path(args.expand_results_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    logger = _setup_pipeline_logger(output_dir=log_root, log_level=args.log_level)

    try:
        run_pipeline(args, logger)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (KeyboardInterrupt).")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed with unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
