import os
import sys
import math
import shutil
from pathlib import Path
import pretty_midi
import mido
from tqdm import tqdm

def list_midi_files(root: Path):
    exts = {".mid", ".midi"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_unique(src: Path, dst_dir: Path):
    safe_mkdir(dst_dir)
    base = src.stem
    ext = src.suffix
    dst = dst_dir / f"{base}{ext}"
    k = 1
    while dst.exists():
        dst = dst_dir / f"{base}__{k}{ext}"
        k += 1
    shutil.copy2(src, dst)
    return dst

def pitch_stats(pm: pretty_midi.PrettyMIDI):
    pitches = []
    for inst in pm.instruments:
        for n in inst.notes:
            pitches.append(n.pitch)
    if not pitches:
        return None
    lo = min(pitches)
    hi = max(pitches)
    span = hi - lo
    return lo, hi, span

def note_overlap_ratio(notes):
    if len(notes) < 2:
        return 0.0
    notes = sorted(notes, key=lambda n: (n.start, n.end))
    overlaps = 0
    total_pairs = 0
    active_end_times = []
    for n in notes:
        active_end_times = [e for e in active_end_times if e > n.start]
        total_pairs += len(active_end_times)
        overlaps += len(active_end_times)
        active_end_times.append(n.end)
    if total_pairs == 0:
        return 0.0
    return overlaps / total_pairs

def monophonic_score(pm: pretty_midi.PrettyMIDI):
    all_notes = []
    for inst in pm.instruments:
        all_notes.extend(inst.notes)
    if len(all_notes) < 20:
        return 0.0
    all_notes = sorted(all_notes, key=lambda n: (n.start, n.end))
    overlap_time = 0.0
    total_time = 0.0
    for i in range(len(all_notes) - 1):
        a = all_notes[i]
        b = all_notes[i + 1]
        total_time += max(0.0, a.end - a.start)
        if b.start < a.end:
            overlap_time += (a.end - b.start)
    total_time += max(0.0, all_notes[-1].end - all_notes[-1].start)
    if total_time <= 1e-9:
        return 0.0
    overlap_frac = overlap_time / total_time
    return max(0.0, 1.0 - min(1.0, overlap_frac * 4.0))

def piano_polyphony_score(pm: pretty_midi.PrettyMIDI):
    piano_notes = []
    piano_inst_count = 0
    for inst in pm.instruments:
        prog = inst.program if not inst.is_drum else None
        is_piano = (prog is not None) and (0 <= prog <= 7)
        if is_piano:
            piano_inst_count += 1
            piano_notes.extend(inst.notes)
    if piano_inst_count == 0 or len(piano_notes) < 30:
        return 0.0
    ov = note_overlap_ratio(piano_notes)
    poly = min(1.0, ov * 3.0)
    density = min(1.0, len(piano_notes) / 600.0)
    return 0.6 * poly + 0.4 * density

def range_extremes_score(pm: pretty_midi.PrettyMIDI):
    ps = pitch_stats(pm)
    if ps is None:
        return 0.0
    lo, hi, span = ps
    span_score = min(1.0, span / 48.0)
    low_score = 1.0 if lo <= 40 else max(0.0, (52 - lo) / 12.0)
    high_score = 1.0 if hi >= 84 else max(0.0, (hi - 72) / 12.0)
    both = min(1.0, (low_score + high_score) / 2.0)
    return 0.55 * span_score + 0.45 * both

def messy_score(midi_path: Path):
    try:
        mf = mido.MidiFile(str(midi_path))
    except Exception:
        return 1.0
    tempo_changes = 0
    time_sigs = 0
    msg_count = 0
    note_on = 0
    weird_delta = 0
    for track in mf.tracks:
        for msg in track:
            msg_count += 1
            if getattr(msg, "time", 0) < 0:
                weird_delta += 1
            if msg.type == "set_tempo":
                tempo_changes += 1
            if msg.type == "time_signature":
                time_sigs += 1
            if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                note_on += 1
    huge = 1.0 if msg_count > 200000 else min(1.0, msg_count / 200000.0)
    too_many_tempos = 1.0 if tempo_changes > 50 else min(1.0, tempo_changes / 50.0)
    too_many_ts = 1.0 if time_sigs > 20 else min(1.0, time_sigs / 20.0)
    dense_notes = 1.0 if note_on > 50000 else min(1.0, note_on / 50000.0)
    weird = 1.0 if weird_delta > 0 else 0.0
    return min(1.0, 0.35 * huge + 0.25 * too_many_tempos + 0.15 * too_many_ts + 0.2 * dense_notes + 0.05 * weird)

def analyze_one(midi_path: Path):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return None
    mono = monophonic_score(pm)
    piano_poly = piano_polyphony_score(pm)
    rng = range_extremes_score(pm)
    messy = messy_score(midi_path)
    return {
        "path": midi_path,
        "mono": mono,
        "piano_poly": piano_poly,
        "range": rng,
        "messy": messy,
    }

def pick_top(scored, key, k):
    scored = sorted(scored, key=lambda x: x[key], reverse=True)
    return scored[:k]

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 select_midis.py /path/to/midis /path/to/OUT")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]).expanduser().resolve()

    if not root.exists():
        print(f"Input folder not found: {root}")
        sys.exit(1)

    files = list(list_midi_files(root))
    if not files:
        print("No MIDI files found.")
        sys.exit(1)

    scored = []
    failures = 0
    for f in tqdm(files, desc="Analyzing MIDIs"):
        r = analyze_one(f)
        if r is None:
            failures += 1
            continue
        scored.append(r)

    safe_mkdir(out)
    safe_mkdir(out / "mono_melody")
    safe_mkdir(out / "piano_polyphony")
    safe_mkdir(out / "range_extremes")
    safe_mkdir(out / "messy")

    mono_sel = [x for x in scored if x["mono"] >= 0.85]
    piano_sel = [x for x in scored if x["piano_poly"] >= 0.55]
    range_sel = [x for x in scored if x["range"] >= 0.65]

    mono_pick = pick_top(mono_sel, "mono", 25) if mono_sel else pick_top(scored, "mono", 25)
    piano_pick = pick_top(piano_sel, "piano_poly", 25) if piano_sel else pick_top(scored, "piano_poly", 25)
    range_pick = pick_top(range_sel, "range", 25) if range_sel else pick_top(scored, "range", 25)

    messy_pick = pick_top(scored, "messy", 2)
    if failures > 0:
        for _ in range(max(0, 2 - len(messy_pick))):
            pass

    used = set()

    def emit(picks, folder):
        for x in picks:
            p = x["path"]
            if p in used:
                continue
            copy_unique(p, out / folder)
            used.add(p)

    emit(mono_pick, "mono_melody")
    emit(piano_pick, "piano_polyphony")
    emit(range_pick, "range_extremes")
    emit(messy_pick, "messy")

    summary = {
        "total_files_found": len(files),
        "parsed_ok": len(scored),
        "parse_failed": failures,
        "mono_exported": len(list((out / "mono_melody").glob("*.mid"))) + len(list((out / "mono_melody").glob("*.midi"))),
        "piano_poly_exported": len(list((out / "piano_polyphony").glob("*.mid"))) + len(list((out / "piano_polyphony").glob("*.midi"))),
        "range_exported": len(list((out / "range_extremes").glob("*.mid"))) + len(list((out / "range_extremes").glob("*.midi"))),
        "messy_exported": len(list((out / "messy").glob("*.mid"))) + len(list((out / "messy").glob("*.midi"))),
    }
    print("\nDone. Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()