# ComfyUI-DukeStereo

A ComfyUI custom node that converts video frames to side-by-side (SBS) stereo format for VR viewing.

## Features

- **Polylines Fill** - High-quality gap filling during stereo warping (from StereoImage Node)
- **Edge-aware depth blur** - Smooths depth while preserving edges
- **Streaming output** - RAM-efficient, writes frames directly to video
- **H.264/yuv420p** - VR-player compatible output
- **Format-Strings** - `%date:yyyy-MM-dd%`, `%time:HH-mm-ss%`, `%divergence%`, `%width%` etc.
- **Home-Expansion** - `~/video/...` paths supported
- **Auto-Increment** - Existing files get `_001`, `_002` suffix

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/killertofu86/ComfyUI-DukeStereo.git
pip install -r ComfyUI-DukeStereo/requirements.txt
```

## Requirements

- torch, transformers, numpy, Pillow, numba, scipy, ffmpeg

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| output_path | stereo_output.mp4 | Output path with format strings |
| divergence | 2.5 | 3D strength (higher = more depth) |
| separation | 0.0 | Additional horizontal offset |
| stereo_offset_exponent | 2.0 | Non-linear depth curve |
| convergence_point | 1.0 | Zero-parallax plane (0.0-1.0) |
| fill_technique | polylines_soft | Gap filling: polylines_soft, polylines_sharp, none |
| depth_blur | 0 | Depth blur kernel (0 = off) |
| depth_blur_edge_threshold | 6.0 | Edge preservation threshold |
| fps | 24 | Output framerate |
| crf | 19 | Quality (lower = better) |

## Inputs

- **images** - Video frames (IMAGE)
- **depth_maps** - Depth maps from LeReS or similar (IMAGE)
- **audio** - Optional ComfyUI AUDIO, e.g. the audio output of VHS Load Video

Both stereo output nodes support audio. Connect the video loader's `audio`
output directly to the stereo node; audio bypasses image upscaling and depth
processing. Without this connection, output remains silent.

The batch node retains the first batch's audio and muxes it once, after the last
meta-batch. Supply the complete source audio, as VHS Load Video does, rather than
individual audio chunks. Audio is encoded as AAC; the H.264 video is copied
without re-encoding. Longer audio is trimmed to the video; shorter audio is
padded with silence. Temporary audio files use the output directory and are
removed after muxing. If muxing fails, the silent video is retained.

For a directory workflow, expose the path loader's AUDIO output from its group
and route audio from the same source as the selected images. Output fps must
match the loader's effective frame rate for synchronization. After updating,
restart ComfyUI before starting a new meta-batch run.

## Output Path Examples

```
~/video/stereo_%date:yyyy-MM-dd%_%time:HH-mm-ss%.mp4
~/video/test_div%divergence%.mp4
/output/stereo_%width%x%height%.mp4
```

## License

MIT

## DA3: automatische Tiefenvorbereitung und Stereo

Nach dem Update ComfyUI neu starten und die Webseite neu laden. Beide Nodes
bleiben aktiv (Modus Always). Ein Queue-Start führt die beiden Durchläufe
automatisch hintereinander aus:

1. VHS Batch Manager an VHS Load Video, **Duke DA3 Prepare Sequence** und
   **Duke Stereo Prepared Sequence** anschließen. Nur dieses Node-Paar an
   diesem Manager betreiben.
2. Die Originalbilder des Loaders direkt an Prepare anschließen. Parallel die
   Bilder über den Upscaler an den IMAGE-Eingang des Stereo-Nodes führen.
   AUDIO des Loaders direkt an Stereo anschließen.
3. **(Down)Load Depth Anything V3 Model** mit Prepare verbinden (Typ
   `DA3MODEL`). Small oder Base mit bf16 und sdpa verwenden. Die STRING-Ausgabe
   `depth_sequence` von Prepare direkt mit Stereo verbinden.
4. Bei Prepare `workflow_mode=prepare_and_stereo` verwenden (Standard).
   Mit VHS-Batches von 16, DA3 `chunk_size=64` und `overlap=8` beginnen.
   `fps` auf die effektive VHS-Bildrate stellen; die Werte für Quelle und
   Frameauswahl in beiden Nodes identisch setzen. Ausgabe- und Cache-Pfad
   müssen beschreibbar sein.
5. Einmal starten. Prepare blockiert den Stereo-Ausgang während der gesamten
   Tiefenvorbereitung. Nach dem letzten Batch wird VHS zurückgesetzt und liest
   dieselbe Quelle mit denselben Einstellungen erneut. Erst dieser zweite
   Durchlauf skaliert die Bilder hoch und schreibt Stereo samt Audio.

Der IMAGE-Eingang am neuen Stereo-Node ist lazy: Der Upscaler wird für diesen
Node erst im zweiten Durchlauf benötigt. Zusätzliche aktive Preview-Ausgaben
am Upscaler können ihn unabhängig davon auch im ersten Durchlauf ausführen.
Für den automatischen Einzelvideo-Ablauf keine Quelle wechseln, bis beide
Durchläufe abgeschlossen sind.

### Verzeichnis automatisch abarbeiten

Für DA3 den **Duke DA3 Directory Queue** anstelle von Loopy Dir einsetzen.
Ein vorhandener GIR-Verzeichnispicker kann weiterhin den Verzeichnispfad liefern.
Die übrigen VHS-/Upscaler-Verbindungen bleiben erhalten:

- Derselbe VHS Batch Manager wird mit Directory Queue, VHS Load Video (Path),
  Prepare und Stereo verbunden.
- `video_path` der Directory Queue wird an `video` von VHS Load Video (Path)
  angeschlossen. Beide Bild-/Audiozweige müssen diesen Loader verwenden.
- Prepare bleibt auf `prepare_and_stereo`; Prepare und Stereo bleiben aktiv.
- `directory` bezeichnet den Eingabeordner, `start_index` ist nullbasiert,
  `max_videos=0` verarbeitet alle verbleibenden Videos. Die Liste wird zu Beginn
  aufgenommen, nach Dateiname sortiert und endet nach dem letzten Video.

Die Queue hält die aktuelle Datei während beider Durchläufe fest. Erst nach
fertigem Stereo und erfolgreichem Audio-Mux folgt das nächste Video. Die
Quellenkennung wird automatisch aus Dateipfad, Größe und Änderungszeit gehasht;
Prepare und Stereo übernehmen sie aus dem gemeinsamen Manager. Persönliche
Quelldateinamen werden nicht im Tiefenmanifest gespeichert. Änderungen an
Quelldateien während des Laufs werden abgewiesen.

Im Verzeichnismodus wird der Tiefencache nach erfolgreicher Ausgabe gelöscht.
Bei Fehlern beim Stereo-Schreiben oder Audio-Mux bleibt der Cache erhalten und
die Queue schaltet nicht weiter. Neue manuelle Läufe starten bei `start_index`;
es gibt keine automatische Wiederaufnahme eines abgebrochenen Prozesses.
Weitere aktive Video-Loader oder Ausgaben benötigen einen eigenen Manager.

Ein vollständiges API-Prompt-Beispiel liegt in
[`examples/da3_directory_api.json`](examples/da3_directory_api.json).
Eingabeverzeichnis, vorhandenes Modell, Bildrate und Ausgabepfade vor Verwendung
anpassen. Das Beispiel verwendet den Duke NCNN Upscaler und enthält keine
Vorschau, die Upscaling im ersten Durchlauf auslösen könnte. `%source_key%` im
Ausgabepfad wird durch die automatische Quellenkennung ersetzt.

Für reine Tiefenvorbereitung gibt es weiterhin `workflow_mode=prepare_only`.
Dabei den Stereo-Node deaktivieren. Eine fertige Folge kann anschließend über
einen STRING-/Primitive-Node am `depth_sequence`-Eingang erneut verwendet werden,
ohne DA3 neu auszuführen.

`source_key` muss für jedes Video eindeutig und in beiden Durchläufen gleich
sein. Ebenso müssen `fps`, `start_frame` und `frame_stride` übereinstimmen.
Die letzten beiden Felder dokumentieren Startframe und Auswahlabstand des
Loaders; sie verändern dessen Auswahl nicht. Alle weiteren Loader-Einstellungen
(z. B. Frame-Limit und erzwungene Bildrate) ebenfalls identisch halten.
Die Kennung ist eine manuelle Zuordnung, keine automatische Inhaltsprüfung.
Im automatischen Modus teilen sich genau ein Prepare- und ein Stereo-Node
denselben Batch Manager; weitere Video-Ausgaben benötigen einen eigenen Ablauf.

Die Datei `manifest.json` markiert eine fertige Folge. Pro Frame wird eine
verlustfreie float32-NPY-Tiefenkarte gespeichert; die gemeinsame globale
Min-/Max-Skalierung erfolgt beim Lesen. Das entspricht DA3 **Standard** ohne
Invertierung, nicht dessen V2-Style-Normalisierung. Nah liegt bei 0, fern bei 1;
`convergence_point` bezeichnet die Ebene ohne Parallaxe. Es findet keine erneute
Min-/Max-Skalierung pro Frame statt. Upscaling darf die räumliche Auflösung
ändern, aber keine Frames entfernen, hinzufügen oder umsortieren.

Bei 723 Frames und etwa 1024² Pixeln sind rund 2,8 GiB für die Tiefe einzuplanen.
Einzelvideo-Folgen bleiben mit `cache_policy=keep` (Standard) für weitere
Stereo-Versuche erhalten. `delete_after_stereo` entfernt sie nach erfolgreichem
Schreiben und Audio-Mux. Im Verzeichnismodus gilt automatisch diese Löschregel.
Es werden ausschließlich entsprechend markierte, von DukeStereo erzeugte
Cache-Verzeichnisse gelöscht; die Ausgabedateien bleiben erhalten. Fehler während der
Vorbereitung entfernen den unvollständigen Cache; VHS-Reset räumt ihn ebenfalls
auf. Nach Prozessabsturz können unvollständige Verzeichnisse ohne Manifest
zurückbleiben. Ein Neustart berechnet neu; eine Fortsetzung ab Checkpoint ist
noch nicht implementiert.

Der Prepare-Fortschritt zeigt Pufferung, Inferenzabschnitte, Ausrichtung und
Ermittlung des gemeinsamen Tiefenbereichs. Der Stereo-Fortschritt zählt die
Frames im zweiten Durchlauf; im Verzeichnismodus wird zusätzlich die
Videoposition protokolliert. VHS selbst zählt Requeues weiterhin über beide
Phasen; dessen Batch-Zähler ist kein Gesamtfortschritt für den DA3-Ablauf.

Pufferung und Überlappung wurden synthetisch auch mit 723 Frames und wechselnden
Batchgrößen geprüft. Der automatische Einzelvideo-Ablauf wurde in ComfyUI mit
Small und Base bei bf16, 64/8, 1024² und 48 FPS durchlaufen: Beide Ausgaben
enthielten 723 decodierbare Frames in 4096 × 2048 SBS mit Audio. Die
Directory-Queue ist mit simulierten Zwei-Video-Läufen inklusive Audiofehler,
Cache-Aufräumen und Quellenwechsel getestet; ein vollständiger Verzeichnislauf
im ComfyUI-Frontend steht noch aus.
