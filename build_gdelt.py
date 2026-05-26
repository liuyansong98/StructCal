from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GdeltEvent:
    head: str
    relation: str
    tail: str
    event_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ICEWS-style temporal KG dataset from GDELT raw data."
    )
    parser.add_argument(
        "--raw-csv",
        default=r".\data\GDELT25\raw\bq-results-20260324-013845-1774316368166.csv",
        help="Path to the raw GDELT CSV.",
    )
    parser.add_argument(
        "--cameo-csv",
        default=r".\data\GDELT25\raw\cameo.csv",
        help="Path to the EventBaseCode -> CAMEO description mapping CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=r".\data\GDELT25",
        help="Directory for the processed dataset.",
    )
    parser.add_argument(
        "--start-date",
        default="20250101",
        help="Inclusive start date in YYYYMMDD.",
    )
    parser.add_argument(
        "--end-date",
        default="20250131",
        help="Inclusive end date in YYYYMMDD.",
    )
    parser.add_argument(
        "--time-field",
        choices=("SQLDATE", "DATEADDED"),
        default="SQLDATE",
        help="Timestamp source for the temporal KG. SQLDATE is day-level; DATEADDED is second-level UTC.",
    )
    parser.add_argument(
        "--time-id-step",
        type=int,
        default=24,
        help="Gap between consecutive timestamp ids. Use 24 for day-level style, 15 for 15-minute granularity style.",
    )
    parser.add_argument(
        "--date-filter-field",
        choices=("SQLDATE", "DATEADDED", "BOTH"),
        default="SQLDATE",
        help="Field used for the inclusive date-window filter. BOTH requires SQLDATE and DATEADDED to both fall inside the range.",
    )
    parser.add_argument(
        "--require-same-sqldate-and-dateadded-day",
        action="store_true",
        help="Require DATEADDED[:8] == SQLDATE for retained events.",
    )
    parser.add_argument(
        "--drop-unseen-in-valid-test",
        action="store_true",
        help="Drop validation/test events whose entities or relations are unseen in train.",
    )
    return parser.parse_args()


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    text = value.strip().replace("\t", " ").replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def yyyymmdd_to_iso(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def yyyymmddhhmmss_to_iso(value: str) -> str:
    return (
        f"{value[:4]}-{value[4:6]}-{value[6:8]} "
        f"{value[8:10]}:{value[10:12]}:{value[12:14]}"
    )


def load_cameo_map(path: Path) -> dict[str, str]:
    cameo_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            code = normalize_text(row[0])
            desc = normalize_text(row[1])
            if not code or not desc:
                continue
            cameo_map[code] = desc
    return cameo_map


def build_relation(event_base_code: str, cameo_map: dict[str, str]) -> str:
    code = normalize_text(event_base_code)
    desc = cameo_map.get(code, "")
    return desc or code


def build_actor(name: str, code: str) -> str:
    return normalize_text(name) or normalize_text(code)


def read_events(
    raw_csv: Path,
    cameo_map: dict[str, str],
    start_date: str,
    end_date: str,
    time_field: str,
    date_filter_field: str,
    require_same_sqldate_and_dateadded_day: bool,
) -> list[GdeltEvent]:
    events: list[GdeltEvent] = []
    with raw_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "SQLDATE",
            "Actor1Code",
            "Actor1Name",
            "Actor2Code",
            "Actor2Name",
            "EventBaseCode",
        }
        if time_field == "DATEADDED" or date_filter_field in ("DATEADDED", "BOTH"):
            required.add("DATEADDED")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"GDELT raw CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            sql_date = normalize_text(row["SQLDATE"])
            dateadded_raw = normalize_text(row.get("DATEADDED", ""))

            dateadded_day = dateadded_raw[:8] if len(dateadded_raw) >= 8 else ""
            if date_filter_field == "DATEADDED":
                if not dateadded_day or dateadded_day < start_date or dateadded_day > end_date:
                    continue
            elif date_filter_field == "SQLDATE":
                if not sql_date or sql_date < start_date or sql_date > end_date:
                    continue
            else:
                if not sql_date or sql_date < start_date or sql_date > end_date:
                    continue
                if not dateadded_day or dateadded_day < start_date or dateadded_day > end_date:
                    continue

            if require_same_sqldate_and_dateadded_day:
                if not sql_date or not dateadded_day or sql_date != dateadded_day:
                    continue

            head = build_actor(row.get("Actor1Name", ""), row.get("Actor1Code", ""))
            tail = build_actor(row.get("Actor2Name", ""), row.get("Actor2Code", ""))
            relation = build_relation(row.get("EventBaseCode", ""), cameo_map)

            if not head or not tail or not relation:
                continue

            if time_field == "DATEADDED":
                raw_time = dateadded_raw
                if len(raw_time) != 14 or not raw_time.isdigit():
                    continue
                event_time = yyyymmddhhmmss_to_iso(raw_time)
            else:
                event_time = yyyymmdd_to_iso(sql_date)

            events.append(
                GdeltEvent(
                    head=head,
                    relation=relation,
                    tail=tail,
                    event_date=event_time,
                )
            )

    events.sort(key=lambda item: (item.event_date, item.head, item.relation, item.tail))
    if not events:
        raise ValueError("No valid GDELT events remained after filtering.")
    return events


def split_chronologically(
    events: list[GdeltEvent], train_ratio: float = 0.8, valid_ratio: float = 0.1
) -> tuple[list[GdeltEvent], list[GdeltEvent], list[GdeltEvent]]:
    if len(events) < 3:
        raise ValueError("Need at least 3 events to create train/valid/test splits.")

    timestamps = []
    counts: dict[str, int] = {}
    for event in events:
        if event.event_date not in counts:
            timestamps.append(event.event_date)
            counts[event.event_date] = 0
        counts[event.event_date] += 1

    if len(timestamps) < 3:
        raise ValueError("Need events from at least 3 timestamps to create train/valid/test splits.")

    total = len(events)
    train_target = total * train_ratio
    valid_target = total * valid_ratio

    train_ts: set[str] = set()
    valid_ts: set[str] = set()
    test_ts: set[str] = set()

    train_count = 0
    valid_count = 0

    for idx, ts in enumerate(timestamps):
        remaining = len(timestamps) - idx
        if train_count < train_target and remaining > 2:
            train_ts.add(ts)
            train_count += counts[ts]
        elif valid_count < valid_target and remaining > 1:
            valid_ts.add(ts)
            valid_count += counts[ts]
        else:
            test_ts.add(ts)

    train = [event for event in events if event.event_date in train_ts]
    valid = [event for event in events if event.event_date in valid_ts]
    test = [event for event in events if event.event_date in test_ts]

    if not train or not valid or not test:
        raise ValueError("Chronological timestamp-group split produced an empty train, valid, or test set.")
    return train, valid, test


def drop_unseen(train: list[GdeltEvent], holdout: list[GdeltEvent]) -> list[GdeltEvent]:
    entities = {item.head for item in train} | {item.tail for item in train}
    relations = {item.relation for item in train}
    return [
        item
        for item in holdout
        if item.head in entities and item.tail in entities and item.relation in relations
    ]


def build_ids(
    events: list[GdeltEvent],
    time_id_step: int,
) -> tuple[OrderedDict[str, int], OrderedDict[str, int], OrderedDict[str, int]]:
    entity_names = sorted({item.head for item in events} | {item.tail for item in events})
    relation_names = sorted({item.relation for item in events})
    all_dates = sorted({item.event_date for item in events})
    entity2id: OrderedDict[str, int] = OrderedDict((name, idx) for idx, name in enumerate(entity_names))
    relation2id: OrderedDict[str, int] = OrderedDict((name, idx) for idx, name in enumerate(relation_names))
    ts2id: OrderedDict[str, int] = OrderedDict((day, idx * time_id_step) for idx, day in enumerate(all_dates))
    return entity2id, relation2id, ts2id


def write_name_id_txt(path: Path, mapping: OrderedDict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        for key, idx in mapping.items():
            f.write(f"{key}\t{idx}\n")


def write_json(path: Path, payload: OrderedDict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def write_split(
    path: Path,
    events: list[GdeltEvent],
    entity2id: OrderedDict[str, int],
    relation2id: OrderedDict[str, int],
    ts2id: OrderedDict[str, int],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        for item in events:
            f.write(
                f"{entity2id[item.head]}\t{relation2id[item.relation]}\t{entity2id[item.tail]}"
                f"\t{ts2id[item.event_date]}\t0\n"
            )


def write_readme(
    path: Path,
    dataset_name: str,
    raw_csv: Path,
    cameo_csv: Path,
    start_date: str,
    end_date: str,
    time_field: str,
    date_filter_field: str,
    require_same_sqldate_and_dateadded_day: bool,
    time_id_step: int,
    all_events: list[GdeltEvent],
    dropped_valid: int,
    dropped_test: int,
) -> None:
    content = [
        f"# {dataset_name}",
        "",
        "This dataset was converted from GDELT raw events into the ICEWS-style temporal KG format used in this repository.",
        "",
        "## Raw Source",
        f"- Raw CSV: {raw_csv.resolve()}",
        f"- CAMEO mapping: {cameo_csv.resolve()}",
        f"- Date filter field: {date_filter_field}",
        f"- Date filter range: {start_date} to {end_date}",
        f"- Require DATEADDED[:8] == SQLDATE: {require_same_sqldate_and_dateadded_day}",
        "- head = Actor1Name, fallback to Actor1Code when Actor1Name is empty",
        "- tail = Actor2Name, fallback to Actor2Code when Actor2Name is empty",
        "- relation = CAMEO description (without the leading EventBaseCode)",
        f"- time = {time_field}",
        "- Split rule: chronological 80% train / 10% valid / 10% test",
        f"- ts2id step = {time_id_step}",
        "",
        "## Coverage",
        f"- First retained date: {all_events[0].event_date}",
        f"- Last retained date: {all_events[-1].event_date}",
        f"- Dropped valid events with unseen train entities/relations: {dropped_valid}",
        f"- Dropped test events with unseen train entities/relations: {dropped_test}",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_csv = Path(args.raw_csv)
    cameo_csv = Path(args.cameo_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameo_map = load_cameo_map(cameo_csv)
    all_events = read_events(
        raw_csv,
        cameo_map,
        args.start_date,
        args.end_date,
        args.time_field,
        args.date_filter_field,
        args.require_same_sqldate_and_dateadded_day,
    )
    train, valid, test = split_chronologically(all_events)

    dropped_valid = 0
    dropped_test = 0
    if args.drop_unseen_in_valid_test:
        raw_valid_len = len(valid)
        raw_test_len = len(test)
        valid = drop_unseen(train, valid)
        test = drop_unseen(train, test)
        dropped_valid = raw_valid_len - len(valid)
        dropped_test = raw_test_len - len(test)

        if not valid or not test:
            raise ValueError("Validation or test became empty after dropping unseen entities/relations.")

    final_events = train + valid + test
    entity2id, relation2id, ts2id = build_ids(final_events, args.time_id_step)

    write_split(output_dir / "train.txt", train, entity2id, relation2id, ts2id)
    write_split(output_dir / "valid.txt", valid, entity2id, relation2id, ts2id)
    write_split(output_dir / "test.txt", test, entity2id, relation2id, ts2id)
    write_split(output_dir / "all_facts.txt", final_events, entity2id, relation2id, ts2id)
    write_name_id_txt(output_dir / "entity2id.txt", entity2id)
    write_name_id_txt(output_dir / "relation2id.txt", relation2id)
    write_json(output_dir / "entity2id.json", entity2id)
    write_json(output_dir / "relation2id.json", relation2id)
    write_json(output_dir / "ts2id.json", ts2id)
    (output_dir / "stat.txt").write_text(
        f"{len(entity2id)}\t{len(relation2id)}\t0\n",
        encoding="utf-8",
    )
    write_readme(
        output_dir / "README.md",
        output_dir.name,
        raw_csv,
        cameo_csv,
        args.start_date,
        args.end_date,
        args.time_field,
        args.date_filter_field,
        args.require_same_sqldate_and_dateadded_day,
        args.time_id_step,
        all_events,
        dropped_valid,
        dropped_test,
    )

    print(f"All retained events: {len(all_events)}")
    print(f"Train/Valid/Test: {len(train)}/{len(valid)}/{len(test)}")
    print(f"Entities: {len(entity2id)}")
    print(f"Relations: {len(relation2id)}")
    print(f"Timestamps: {len(ts2id)}")
    print(f"Dropped valid unseen: {dropped_valid}")
    print(f"Dropped test unseen: {dropped_test}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
