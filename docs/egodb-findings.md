# EgoVerse registry — what's actually in there

Measured live on 2026-08-15 against `app.episodes` via `src/egodb.py`.
Reproduce: `AWS_PROFILE=egoverse .venv/bin/python src/egodb.py`

## Access (no `egomimic` install needed)

```bash
aws configure set aws_access_key_id <key> --profile egoverse   # published in EgoVerse README
aws configure set region us-east-2 --profile egoverse
AWS_PROFILE=egoverse bash external/EgoVerse/egomimic/utils/aws/setup_secret.sh
```

The README's AWS key belongs to IAM user `egoverse-public` and only bootstraps
Secrets Manager, which hands back **read-only** Postgres + R2 credentials into
`~/.egoverse_env`. `src/egodb.py` reads that file directly, so none of the training
stack (torch, mujoco-py, ros, projectaria-tools) has to install.

## Scale: the live DB is ~5.5× the paper

| | Paper (v2) | Live DB |
|---|---|---|
| Episodes | 80,000 | **439,053** active (+7,904 soft-deleted) |
| Hours | 1,362 | **~4,003** (432M frames @30fps) |
| Tasks | 1,965 | **27,997** |
| Scenes | 240 | 402 |
| Demonstrators | 2,087 | 4,588 `operator` values |

## One contributor is now 81% of the dataset

| lab | episodes |
|---|---|
| **microagi** | 355,926 |
| mecka | 41,617 |
| abc | 18,456 |
| scale | 17,090 |
| rl2 | 4,811 |
| eth / song / yam / wang | 543 / 281 / 182 / 147 |

`microagi` and `abc` are **not in the paper's author affiliations**. microagi alone
outweighs every paper-era contributor combined by ~8×, and it supplies no `scene`,
no `operator`, and no `rig_name`.

## Metadata sparsity — the "what's sparse" answer

| Field | Populated |
|---|---|
| `scene` | 13.4% (380,338 empty) |
| `operator` | 18.9% (355,926 empty) |
| `segments` (annotations) | **5.3%** (23,289 of 439,053) |
| `rig_name` | 18.9% |

**This matters:** the paper's headline finding is that *scene* and *demonstrator*
diversity drive generalization unevenly. Both axes are missing for ~81–87% of the
current dataset. Any diversity score that relies on metadata alone covers a small
minority of episodes — which is itself a reportable result.

`segments` format is `[{"label": str, "start_seconds": float, "end_seconds": float}]`
— **seconds**, whereas the Zarr annotations use frame indices (`start_idx`/`end_idx`).
Different units in the two stores; don't mix them.

## `eval_success` is uniformly useless

| Field | Value |
|---|---|
| `is_eval` | **False for all 439,053 rows** |
| `eval_success` | `True` for all 420,415 non-null — never set, just the dataclass default |
| `eval_score` | only `-1.0` / NaN |

There is **no success/failure ground truth in the registry.**

### …but outcome labels are hidden in task names

1,586 rl2 episodes encode the outcome in `task` instead:

| task | n |
|---|---|
| `cup_on_saucer_success` + `_success_1` | 748 |
| `cup_on_saucer_failure` + `_failure_1` | 165 |
| `fold_clothes_success` + `_success_1` | 321 |
| `bag_groceries_success` (+`_2`,`_3`) | 347 |
| `bag_groceries_failure` | 5 |

**`cup_on_saucer` is the usable one: 748 success vs 165 failure, one task, one lab.**
That is a real labelled set for a success classifier — invisible to anyone who checks
the `eval_success` column and moves on.
(Beware regex false positives: `baked_goods`, `packaged_goods` match `/good/`.)

## Task naming has decayed since the paper

27,997 task names for 439k episodes. The taxonomy has duplicates and near-synonyms:

- `ironing_clothes` (3,800) vs `iron_clothes` (2,353) — same concept, two names
- `fold_clothes` (12,617) vs `fold_laundry` (9,044)
- `prepare_food` / `prepare_meal` / `prepare_salad` / `prepare_vegetables` / `prepare_potatoes`
- 9,236 singleton task names (33% of all names, 2.1% of episodes), e.g.
  `test_exit_sign`, `repair_wire_fence`, `reattaching_heatsinks`

`CONTRIBUTING_DATA.md` §on `task` explicitly warns against this — "canonicalize your
new `task_name`… not a one-off trial description" — so the guidance exists and is
being ignored at scale. Top 10 tasks cover only 21% of episodes.

## The README's example download command matches 0 episodes

```bash
# README / sync_s3.py preset "aria-fold-clothes":
#   lambda row: row.get('embodiment') == 'aria'
```

`embodiment` now only holds `human_bimanual` (414,511), `eva_bimanual` (21,594),
`human_right_arm` (2,053), `human_left_arm` (612), `yam_bimanual` (182),
`eva_right_arm` (101). **No row contains "aria".** The 07/08 changelog moved vendor
labels out of `embodiment`; the presets were never updated.

Working equivalent: `rig_name == 'aria_gen1' and task == 'fold_clothes'` → **711 episodes**.

## Cheap visual access

Every row carries `zarr_mp4_path` (e.g. `s3://rldb/processed_v3/microagi/<hash>.mp4`)
alongside `zarr_processed_path`. Preview MP4s are far cheaper to pull than full Zarr
stores — use them for anything visual that doesn't need poses.
