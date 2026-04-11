import argparse
import csv
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mido
import pretty_midi
from tqdm import tqdm


MIDI_EXTENSIONS = {".mid", ".midi"}
ROOT_NOTE_RE = re.compile(r"^[A-G](?:b|#)?")


def list_midi_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in MIDI_EXTENSIONS:
            yield path


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_unique(src: Path, dst_dir: Path):
    safe_mkdir(dst_dir)
    base = src.stem
    ext = src.suffix
    dst = dst_dir / f"{base}{ext}"
    suffix = 1
    while dst.exists():
        dst = dst_dir / f"{base}__{suffix}{ext}"
        suffix += 1
    shutil.copy2(src, dst)
    return dst


def copy_relative(src: Path, root: Path, dst_root: Path):
    dst = dst_root / src.relative_to(root)
    safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def pitch_stats(pm: pretty_midi.PrettyMIDI):
    pitches = []
    for inst in pm.instruments:
        for note in inst.notes:
            pitches.append(note.pitch)
    if not pitches:
        return None
    low = min(pitches)
    high = max(pitches)
    return low, high, high - low


def note_overlap_ratio(notes):
    if len(notes) < 2:
        return 0.0
    notes = sorted(notes, key=lambda note: (note.start, note.end))
    overlaps = 0
    total_pairs = 0
    active_end_times = []
    for note in notes:
        active_end_times = [end for end in active_end_times if end > note.start]
        total_pairs += len(active_end_times)
        overlaps += len(active_end_times)
        active_end_times.append(note.end)
    if total_pairs == 0:
        return 0.0
    return overlaps / total_pairs


def monophonic_score(pm: pretty_midi.PrettyMIDI):
    all_notes = []
    for inst in pm.instruments:
        all_notes.extend(inst.notes)
    if len(all_notes) < 20:
        return 0.0
    all_notes = sorted(all_notes, key=lambda note: (note.start, note.end))
    overlap_time = 0.0
    total_time = 0.0
    for idx in range(len(all_notes) - 1):
        current = all_notes[idx]
        nxt = all_notes[idx + 1]
        total_time += max(0.0, current.end - current.start)
        if nxt.start < current.end:
            overlap_time += current.end - nxt.start
    total_time += max(0.0, all_notes[-1].end - all_notes[-1].start)
    if total_time <= 1e-9:
        return 0.0
    overlap_fraction = overlap_time / total_time
    return max(0.0, 1.0 - min(1.0, overlap_fraction * 4.0))


def piano_polyphony_score(pm: pretty_midi.PrettyMIDI):
    piano_notes = []
    piano_inst_count = 0
    for inst in pm.instruments:
        program = inst.program if not inst.is_drum else None
        is_piano = (program is not None) and (0 <= program <= 7)
        if is_piano:
            piano_inst_count += 1
            piano_notes.extend(inst.notes)
    if piano_inst_count == 0 or len(piano_notes) < 30:
        return 0.0
    overlap = note_overlap_ratio(piano_notes)
    polyphony = min(1.0, overlap * 3.0)
    density = min(1.0, len(piano_notes) / 600.0)
    return 0.6 * polyphony + 0.4 * density


def range_extremes_score(pm: pretty_midi.PrettyMIDI):
    stats = pitch_stats(pm)
    if stats is None:
        return 0.0
    low, high, span = stats
    span_score = min(1.0, span / 48.0)
    low_score = 1.0 if low <= 40 else max(0.0, (52 - low) / 12.0)
    high_score = 1.0 if high >= 84 else max(0.0, (high - 72) / 12.0)
    both_extremes = min(1.0, (low_score + high_score) / 2.0)
    return 0.55 * span_score + 0.45 * both_extremes


def messy_score(midi_path: Path):
    try:
        mf = mido.MidiFile(str(midi_path))
    except Exception:
        return 1.0

    tempo_changes = 0
    time_signatures = 0
    message_count = 0
    note_on_count = 0
    weird_delta = 0

    for track in mf.tracks:
        for msg in track:
            message_count += 1
            if getattr(msg, "time", 0) < 0:
                weird_delta += 1
            if msg.type == "set_tempo":
                tempo_changes += 1
            if msg.type == "time_signature":
                time_signatures += 1
            if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                note_on_count += 1

    huge = 1.0 if message_count > 200000 else min(1.0, message_count / 200000.0)
    too_many_tempos = 1.0 if tempo_changes > 50 else min(1.0, tempo_changes / 50.0)
    too_many_time_signatures = 1.0 if time_signatures > 20 else min(1.0, time_signatures / 20.0)
    dense_notes = 1.0 if note_on_count > 50000 else min(1.0, note_on_count / 50000.0)
    weird = 1.0 if weird_delta > 0 else 0.0
    return min(1.0, 0.35 * huge + 0.25 * too_many_tempos + 0.15 * too_many_time_signatures + 0.2 * dense_notes + 0.05 * weird)


def analyze_one(midi_path: Path):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return None
    return {
        "path": midi_path,
        "mono": monophonic_score(pm),
        "piano_poly": piano_polyphony_score(pm),
        "range": range_extremes_score(pm),
        "messy": messy_score(midi_path),
    }


def pick_top(scored, key, limit):
    return sorted(scored, key=lambda row: row[key], reverse=True)[:limit]


def detect_mode(root: Path, requested_mode: str):
    if requested_mode != "auto":
        return requested_mode
    if root.name.startswith("free-midi-chords"):
        return "free-midi-chords"
    return "generic"


def parse_key_folder(folder_name: str):
    parts = folder_name.split(" - ", 2)
    if len(parts) != 3:
        return {"key_index": None, "major_key": "", "minor_key": ""}
    try:
        key_index = int(parts[0])
    except ValueError:
        key_index = None
    return {
        "key_index": key_index,
        "major_key": parts[1],
        "minor_key": parts[2],
    }


def normalize_chord_symbol(chord_symbol: str):
    quality = ROOT_NOTE_RE.sub("", chord_symbol, count=1)
    return quality or "maj"


def parse_free_midi_record(root: Path, midi_path: Path):
    rel = midi_path.relative_to(root)
    parts = rel.parts
    if len(parts) < 3:
        return None

    key_folder = parts[0]
    collection_folder = parts[1]
    key_meta = parse_key_folder(key_folder)

    record = {
        "path": midi_path,
        "relative_path": str(rel),
        "key_folder": key_folder,
        "key_index": key_meta["key_index"],
        "major_key": key_meta["major_key"],
        "minor_key": key_meta["minor_key"],
        "collection_folder": collection_folder,
    }

    if collection_folder == "4 Progression":
        if len(parts) == 4:
            scale_family, filename = parts[2], parts[3]
            style = "straight"
        elif len(parts) == 5:
            scale_family, style_folder, filename = parts[2], parts[3], parts[4]
            style = style_folder.replace(" style", "")
        else:
            return None

        stem_parts = Path(filename).stem.split(" - ")
        if len(stem_parts) != 3:
            return None
        tonic_hint, harmonic_pattern, mood_label = stem_parts
        mood_tags = "|".join(mood_label.split())

        record.update({
            "subset_mode": "free-midi-chords",
            "scale_family": scale_family,
            "style": style,
            "tonic_hint": tonic_hint,
            "pattern_label": harmonic_pattern,
            "surface_label": mood_label,
            "mood_tags": mood_tags,
            "normalized_group_id": f"progression|{scale_family}|{style}|{harmonic_pattern}|{mood_label}",
        })
        return record

    if collection_folder == "3 All chords" and len(parts) == 3:
        scale_family, filename = "mixed", parts[2]
    elif len(parts) == 4:
        scale_family, filename = parts[2], parts[3]
    else:
        return None

    stem_parts = Path(filename).stem.split(" - ", 1)
    if len(stem_parts) != 2:
        return None
    roman_label, chord_symbol = stem_parts
    chord_quality = normalize_chord_symbol(chord_symbol)

    record.update({
        "subset_mode": "free-midi-chords",
        "scale_family": scale_family,
        "style": "straight",
        "tonic_hint": "",
        "pattern_label": roman_label,
        "surface_label": chord_symbol,
        "mood_tags": "",
        "normalized_group_id": f"chord|{collection_folder}|{scale_family}|{roman_label}|{chord_quality}",
    })
    return record


def evenly_spaced_indices(count: int, wanted: int):
    if wanted >= count:
        return list(range(count))
    if wanted <= 1:
        return [count // 2]

    indices = sorted({round(idx * (count - 1) / (wanted - 1)) for idx in range(wanted)})
    next_candidate = 0
    while len(indices) < wanted:
        if next_candidate not in indices:
            indices.append(next_candidate)
        next_candidate += 1
    return sorted(indices)


def write_manifest(records, manifest_path: Path):
    safe_mkdir(manifest_path.parent)
    fieldnames = [
        "selected_path",
        "relative_path",
        "collection_folder",
        "scale_family",
        "style",
        "pattern_label",
        "surface_label",
        "mood_tags",
        "key_folder",
        "key_index",
        "major_key",
        "minor_key",
        "normalized_group_id",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})


def run_generic_selection(root: Path, out: Path):
    files = list(list_midi_files(root))
    if not files:
        print("No MIDI files found.")
        sys.exit(1)

    scored = []
    failures = 0
    for midi_file in tqdm(files, desc="Analyzing MIDIs"):
        result = analyze_one(midi_file)
        if result is None:
            failures += 1
            continue
        scored.append(result)

    safe_mkdir(out)
    safe_mkdir(out / "mono_melody")
    safe_mkdir(out / "piano_polyphony")
    safe_mkdir(out / "range_extremes")
    safe_mkdir(out / "messy")

    mono_sel = [row for row in scored if row["mono"] >= 0.85]
    piano_sel = [row for row in scored if row["piano_poly"] >= 0.55]
    range_sel = [row for row in scored if row["range"] >= 0.65]

    mono_pick = pick_top(mono_sel, "mono", 25) if mono_sel else pick_top(scored, "mono", 25)
    piano_pick = pick_top(piano_sel, "piano_poly", 25) if piano_sel else pick_top(scored, "piano_poly", 25)
    range_pick = pick_top(range_sel, "range", 25) if range_sel else pick_top(scored, "range", 25)
    messy_pick = pick_top(scored, "messy", 2)

    used = set()

    def emit(picks, folder_name):
        for row in picks:
            midi_path = row["path"]
            if midi_path in used:
                continue
            copy_unique(midi_path, out / folder_name)
            used.add(midi_path)

    emit(mono_pick, "mono_melody")
    emit(piano_pick, "piano_polyphony")
    emit(range_pick, "range_extremes")
    emit(messy_pick, "messy")

    summary = {
        "mode": "generic",
        "total_files_found": len(files),
        "parsed_ok": len(scored),
        "parse_failed": failures,
        "mono_exported": len(list((out / "mono_melody").glob("*.mid"))) + len(list((out / "mono_melody").glob("*.midi"))),
        "piano_poly_exported": len(list((out / "piano_polyphony").glob("*.mid"))) + len(list((out / "piano_polyphony").glob("*.midi"))),
        "range_exported": len(list((out / "range_extremes").glob("*.mid"))) + len(list((out / "range_extremes").glob("*.midi"))),
        "messy_exported": len(list((out / "messy").glob("*.mid"))) + len(list((out / "messy").glob("*.midi"))),
    }

    print("\nDone. Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


def run_free_midi_selection(root: Path, out: Path, keys_per_pattern: int):
    files = list(list_midi_files(root))
    if not files:
        print("No MIDI files found.")
        sys.exit(1)

    records = []
    skipped = 0
    for midi_file in tqdm(files, desc="Indexing free-midi-chords"):
        record = parse_free_midi_record(root, midi_file)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    groups = defaultdict(list)
    for record in records:
        groups[record["normalized_group_id"]].append(record)

    selected = []
    for group_id in sorted(groups):
        group_records = sorted(
            groups[group_id],
            key=lambda row: (
                row["key_index"] if row["key_index"] is not None else 999,
                row["relative_path"],
            ),
        )
        for idx in evenly_spaced_indices(len(group_records), keys_per_pattern):
            selected.append(group_records[idx])

    safe_mkdir(out)
    for record in tqdm(selected, desc="Copying selected subset"):
        dst = copy_relative(record["path"], root, out)
        record["selected_path"] = str(dst)

    manifest_path = out / "subset_manifest.csv"
    write_manifest(selected, manifest_path)

    collection_counts = Counter(record["collection_folder"] for record in selected)
    style_counts = Counter(record["style"] for record in selected)

    summary = {
        "mode": "free-midi-chords",
        "total_files_found": len(files),
        "parsed_ok": len(records),
        "parse_skipped": skipped,
        "unique_pattern_groups": len(groups),
        "keys_per_pattern": keys_per_pattern,
        "selected_files": len(selected),
        "manifest_path": str(manifest_path),
    }

    print("\nDone. Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("selected_by_collection:")
    for key, value in sorted(collection_counts.items()):
        print(f"  {key}: {value}")

    print("selected_by_style:")
    for key, value in sorted(style_counts.items()):
        print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Select representative MIDI subsets.")
    parser.add_argument("input_root", help="Input folder containing MIDI files.")
    parser.add_argument("output_root", help="Output folder for the selected subset.")
    parser.add_argument(
        "--mode",
        choices=["auto", "generic", "free-midi-chords"],
        default="auto",
        help="Selection strategy. Auto detects free-midi-chords by folder name.",
    )
    parser.add_argument(
        "--keys-per-pattern",
        type=int,
        default=3,
        help="For free-midi-chords mode, keep this many evenly spaced keys per normalized pattern.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.input_root).expanduser().resolve()
    out = Path(args.output_root).expanduser().resolve()

    if not root.exists():
        print(f"Input folder not found: {root}")
        sys.exit(1)

    mode = detect_mode(root, args.mode)
    if mode == "free-midi-chords":
        run_free_midi_selection(root, out, max(1, args.keys_per_pattern))
        return
    run_generic_selection(root, out)


if __name__ == "__main__":
    main()
