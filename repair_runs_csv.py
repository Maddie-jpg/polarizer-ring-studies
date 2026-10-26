"""
repair_runs_csv.py
==================

One-off conversion of a runs.csv written by the old tolerance_scan.py.

It does two things:

  1. Un-shifts the columns, if they need it.
     The old script appended each machine as its own one-row DataFrame and
     wrote the header only for the first.  A machine that FAILED had different
     keys from one that succeeded, so if the first machine written happened to
     fail, its 10 keys became the file's header and every later 17-value row
     went underneath with no header - putting every value in the wrong column.
     Nothing is lost; the values are all there, in order.

  2. Converts to the column set the new tolerance_scan.py writes.
     The new script no longer tracks injection, so 'inj_survival' goes and
     'error' is always present.  An old file left as it is would have new rows
     appended under the old header, which recreates exactly the bug in (1).

The original is kept as runs_before_repair.csv and runs.csv is replaced, so
the new tolerance_scan.py can resume straight from it.

    python repair_runs_csv.py Results/.../ToleranceScan/ilc_dr/runs.csv
"""
import csv
import os
import shutil
import sys

# What the OLD script wrote for a machine that succeeded (17 values)...
OLD_SUCCESS = [
    'scale', 'seed', 'ok', 'delta_max', 'inj_survival',
    'qx', 'qy', 'orbit_rms_x', 'orbit_rms_y',
    'DA_ratio_min', 'DA_x_pos', 'DA_x_neg', 'DA_y', 'DA_boundary',
    'MA_pos', 'MA_neg', 'runtime_s',
]
# ...and for one that failed (10 values).
OLD_FAILURE = [
    'scale', 'seed', 'ok', 'delta_max', 'inj_survival',
    'DA_ratio_min', 'MA_pos', 'MA_neg', 'error', 'runtime_s',
]
# What the NEW script writes.  Must match RESULT_COLUMNS in tolerance_scan.py.
NEW_COLUMNS = [
    'scale', 'seed', 'ok', 'delta_max',
    'qx', 'qy', 'orbit_rms_x', 'orbit_rms_y',
    'DA_ratio_min', 'DA_x_pos', 'DA_x_neg', 'DA_y', 'DA_boundary',
    'MA_pos', 'MA_neg', 'runtime_s', 'error',
]


def repair(path):
    with open(path, newline='') as f:
        lines = list(csv.reader(f))
    if not lines:
        sys.exit(f'{path} is empty')

    header, body = lines[0], lines[1:]
    print(f'{path}')
    print(f'  header has {len(header)} names, {len(body)} data rows')

    if header == NEW_COLUMNS:
        print('  already in the new format - nothing to do')
        return

    shifted = (header != OLD_SUCCESS)
    print('  columns are shifted - un-shifting' if shifted
          else '  columns are fine - converting to the new format')

    rows, skipped = [], 0
    for values in body:
        if len(values) == len(OLD_SUCCESS):
            row = dict(zip(OLD_SUCCESS, values))
        elif len(values) == len(OLD_FAILURE):
            row = dict(zip(OLD_FAILURE, values))
        else:
            skipped += 1
            continue
        row.pop('inj_survival', None)          # the new script does not track it
        rows.append({c: row.get(c, '') for c in NEW_COLUMNS})

    backup = os.path.join(os.path.dirname(path), 'runs_before_repair.csv')
    shutil.copy2(path, backup)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=NEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(r['ok'] == 'True' for r in rows)
    print(f'  wrote {len(rows)} rows ({n_ok} successful) back to runs.csv')
    print(f'  original kept as {os.path.basename(backup)}')
    if skipped:
        print(f'  skipped {skipped} row(s) with an unexpected number of fields')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        repair(p)
