#!/usr/bin/env python3
"""公開BGMの音源を一括解析し、カタログ全体の一貫性を点検する。

作業用BGMチャンネルでは複数曲が連続再生されるため、曲ごとの
ラウドネス・テンポ・明るさのばらつきがそのまま離脱要因になる。
このスクリプトは1曲ごとの指標と、カタログ全体のばらつきを出す。

使い方:
    python3 tools/analyze_audio.py 音源ディレクトリ_または_ファイル...
"""
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
# 配信基準のターゲット（Spotify/YouTube はおおむね -14 LUFS で正規化）
TARGET_LUFS = -14.0


def ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_to_array(path):
    """任意の音声ファイルを 44.1kHz float32 のステレオ配列へデコードする。"""
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "decoded.wav")
        subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-y", "-i", path,
             "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", wav],
            check=True,
        )
        with wave.open(wav, "rb") as w:
            frames = w.readframes(w.getnframes())
            rate = w.getframerate()
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, 2), rate


def analyze(path):
    import librosa
    import pyloudnorm as pyln

    stereo, rate = decode_to_array(path)
    mono = stereo.mean(axis=1)
    duration = len(mono) / rate

    meter = pyln.Meter(rate)
    lufs = meter.integrated_loudness(stereo)

    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
    # 0dBFS に張り付いたサンプル＝クリッピングの疑い
    clipped = int(np.sum(np.abs(stereo) >= 0.999))

    tempo, _ = librosa.beat.beat_track(y=mono, sr=rate)
    tempo = float(np.atleast_1d(tempo)[0])

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=rate)))

    left, right = stereo[:, 0], stereo[:, 1]
    if np.std(left) > 0 and np.std(right) > 0:
        correlation = float(np.corrcoef(left, right)[0, 1])
    else:
        correlation = 1.0

    chroma = librosa.feature.chroma_cqt(y=mono, sr=rate)
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key = keys[int(np.argmax(np.mean(chroma, axis=1)))]

    return {
        "file": os.path.basename(path),
        "duration": duration,
        "lufs": lufs,
        "peak_db": peak_db,
        "clipped": clipped,
        "tempo": tempo,
        "centroid": centroid,
        "correlation": correlation,
        "key": key,
    }


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in sorted(names)
                          if os.path.splitext(n)[1].lower() in AUDIO_EXT]
        elif os.path.splitext(p)[1].lower() in AUDIO_EXT:
            files.append(p)
    return files


def main(argv):
    files = collect(argv)
    if not files:
        print("音声ファイルが見つかりません。ディレクトリかファイルを指定してください。")
        return 1

    rows = []
    for path in files:
        try:
            rows.append(analyze(path))
        except Exception as exc:
            print(f"!! {os.path.basename(path)}: {type(exc).__name__}: {exc}")

    if not rows:
        return 1

    print(f"\n{'ファイル':<34}{'長さ':>8}{'LUFS':>8}{'ピーク':>8}{'BPM':>7}{'明るさHz':>10}{'相関':>7}{'キー':>5}{'クリップ':>8}")
    print("-" * 95)
    for r in rows:
        m, s = divmod(int(r["duration"]), 60)
        print(f"{r['file'][:33]:<34}{m:>5}:{s:02d}{r['lufs']:>8.1f}"
              f"{r['peak_db']:>8.1f}{r['tempo']:>7.1f}{r['centroid']:>10.0f}"
              f"{r['correlation']:>7.2f}{r['key']:>5}{r['clipped']:>8}")

    lufs = np.array([r["lufs"] for r in rows])
    tempo = np.array([r["tempo"] for r in rows])
    centroid = np.array([r["centroid"] for r in rows])

    print("\n── カタログ全体の一貫性 ──")
    print(f"ラウドネス   平均 {lufs.mean():6.1f} LUFS / 幅 {lufs.max() - lufs.min():.1f} dB"
          f"  （連続再生するなら幅 1.0 dB 以内が目安）")
    print(f"テンポ       平均 {tempo.mean():6.1f} BPM  / 幅 {tempo.max() - tempo.min():.1f} BPM")
    print(f"明るさ       平均 {centroid.mean():6.0f} Hz   / 幅 {centroid.max() - centroid.min():.0f} Hz")

    print("\n── 所見 ──")
    spread = lufs.max() - lufs.min()
    if spread > 1.0:
        loudest = rows[int(np.argmax(lufs))]
        quietest = rows[int(np.argmin(lufs))]
        print(f"・ラウドネスが {spread:.1f} dB ばらついている。"
              f"最大 {loudest['file']}、最小 {quietest['file']}。連続再生で音量差が出る")
    else:
        print("・ラウドネスは揃っている。連続再生に耐える")

    off_target = [r for r in rows if abs(r["lufs"] - TARGET_LUFS) > 1.5]
    if off_target:
        print(f"・配信基準 {TARGET_LUFS} LUFS から 1.5dB 以上外れている曲が {len(off_target)} 件。"
              "プラットフォーム側の正規化で意図した音圧にならない")

    clipping = [r for r in rows if r["clipped"] > 0]
    if clipping:
        print(f"・クリッピングの疑いがある曲が {len(clipping)} 件："
              + "、".join(r["file"] for r in clipping))

    narrow = [r for r in rows if r["correlation"] > 0.95]
    if narrow:
        print(f"・ステレオ幅がほぼモノラルの曲が {len(narrow)} 件。"
              "ヘッドホン前提の作業用BGMでは広がりが効く")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
