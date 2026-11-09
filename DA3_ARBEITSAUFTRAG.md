# Arbeitsauftrag: zeitlich konsistente DA3-Tiefe für Stereo-Videos

## Ziel

Den bestehenden Stereo-Workflow um eine speichersparende Tiefenvorberechnung mit Depth Anything 3 und einen separaten Stereo-Node für vorbereitete Tiefenfolgen ergänzen. Die vorhandenen Nodes und Workflows sollen weiter funktionieren. Neue Nodes gehören in dieses Repository; die installierte `ComfyUI-DepthAnythingV3`-Erweiterung und VideoHelperSuite (VHS) sollen nach Möglichkeit wiederverwendet werden.

## Gewünschter Ablauf

1. Das Originalvideo über VHS-Meta-Batches abschnittsweise einlesen und DA3-Tiefe vorab für den ganzen Clip berechnen. Benachbarte Abschnitte sollen sich überlappen und anhand der gemeinsamen Frames ausgerichtet und im Übergang verblendet werden.
2. Zwischenstände auf Platte ablegen, sodass der RAM-Bedarf nicht mit dem gesamten Video und seinen vollständigen Ergebnisstapeln wächst. Nach dem letzten Batch eine gemeinsame, für Stereo geeignete Tiefenskalierung bestimmen und eine Tiefenfolge speichern.
3. In einem zweiten Durchlauf die Originalframes über den bestehenden Meta-Batch- und Upscaler-Pfad verarbeiten. Ein neuer, separater Stereo-Node liest dazu die passenden vorbereiteten Tiefenframes und schreibt das SBS-Video. Audio und bisherige Ausgabeparameter sollen erhalten bleiben. Der bestehende Stereo-Node wird nicht überladen oder ersetzt.

Der neue Stereo-Pfad muss die gemeinsame Skalierung der Tiefenfolge erhalten: `apply_stereo_divergence` normalisiert bisher jede Tiefenkarte anhand ihres eigenen Minimums und Maximums. Für die vorbereitete Folge darf diese frameweise Neunormalisierung die zeitliche Konsistenz nicht wieder zerstören. Nähe/Ferne, Konvergenz und Tiefenbereich anhand eines tatsächlichen Ergebnisses überprüfen.

## Technische Prüfung und Umsetzung

- VHS-Meta-Batch-Lebenszyklus, Requeue, Abbruch, letzter unvollständiger Batch und Aufräumen temporärer Dateien berücksichtigen. Die Zuordnung von Originalvideo, Tiefenfolge und Audio muss bei mehreren Videos stimmen.
- 64 Frames **insgesamt** pro DA3-Abschnitt mit beispielsweise 8 Frames Überlappung als Ausgangspunkt untersuchen; nach dem ersten Abschnitt entsprechen das 56 neuen Frames pro Schritt. VHS liefert regulär feste Batchgrößen: Pufferung und Abschluss so umsetzen, dass keine Frames fehlen, doppelt ausgegeben oder falsch zugeordnet werden. Abschnittsgröße und Überlappung sollen zur verfügbaren Hardware passen.
- Bestehende DA3-Logik für Multi-View-Inferenz, Überlappung, Sim(3)-Ausrichtung und Verblendung untersuchen und wenn sinnvoll wiederverwenden. Die heutige DA3-Streaming-Node lädt das gesamte VIDEO und hält mehrere große Ergebnisstapel im RAM; bloßes Aufrufen dieser Node pro VHS-Batch erhält keine globale Konsistenz. Öffentliche und private Schnittstellen und ihre Abhängigkeiten vor der Integration prüfen.
- Die aktuelle DA3-Node nutzt gemeinsame Normalisierung über alle ausgerichteten Frames. Eine äquivalente speicherbegrenzte Lösung entwerfen; die Skalierung darf nicht je Batch unabhängig erfolgen. Tiefenausgabe und temporäres Format mit ausreichender Präzision und ohne unnötige verlustbehaftete Kompression wählen.
- Frameanzahl, Reihenfolge, Bildrate, Startposition, Größenanpassung sowie die letzte Teilmenge beider Durchläufe prüfen. Getestetes Video: 723 Frames, 48 FPS, 1024 × 1024. Für Tests vorhandenes Videomaterial aus `homemade/tmp` verwenden; keine persönlichen Inhalte oder Dateinamen in Repo, Logs oder Testdaten übernehmen.
- Den vorhandenen Importpfad zur VHS-Hilfe im Stereo-Node prüfen: Ein Fallback verweist auf `custom_nodes/comfyui-videohelpersuite`, die ältere Kopie wurde deaktiviert; aktiv ist `custom_nodes/ComfyUI-VideoHelperSuite`.
- Keine neuen Downloads ohne konkrete Zustimmung. Vorhandene Small- und Base-Gewichte liegen in `models/depthanything3` der ComfyUI-Installation.

## Bisherige Messungen

GPU: AMD Radeon RX 7900 XTX, 24 GiB VRAM; Rechner: 31 GiB RAM. DA3 Streaming wurde mit 72 Frames bei 1024 × 1024, 64 Frames pro Abschnitt, 8 Frames Überlappung, `bf16` und Torch-Ausrichtung getestet. Die Zeiten enthalten Modellaufbau, Ausrichtung und Schreiben der NPZ-Dateien:

| Modell | Zeit | Spitzen-VRAM | Spitzen-RAM des Prozesses |
| --- | ---: | ---: | ---: |
| Small | 67,9 s | 6,94 GiB | 6,60 GiB |
| Base | 108,0 s | 12,35 GiB | 6,63 GiB |

Large mit 32er-Abschnitten wurde wegen sehr langer Laufzeit auf Wunsch abgebrochen, bevor der erste Abschnitt fertig war. Keine belastbare Large-Zeit oder VRAM-Spitze vorhanden. Small oder Base sind für die Umsetzung vorzuziehen; keine Qualitätsbewertung der Tiefenbilder vorgenommen. Für den Einzelbildpfad vorher gemessene Zeiten sind nicht auf Multi-View-Streaming übertragbar.

## Abnahme

Einen vollständigen Clip aus dem vorhandenen Material durch beide Phasen verarbeiten. Überlappung und Batchgrenzen, Anzahl und Synchronität der Frames, Audio, normalisierte Tiefenwerte und sichtbar ruhiges Stereo prüfen. RAM/VRAM, Laufzeit und Plattenbedarf erfassen, einschließlich Abbruch und Wiederanlauf. Frühere Workflows mit dem bestehenden Stereo-Node müssen unverändert ausführbar bleiben.
