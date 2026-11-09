# TODO

- [x] **Depth-Eingang oben am Stereo-Node:** fester STRING-Anschluss direkt hinter IMAGE; keine Tiefenpfad-Widgets zwischen den übrigen Einstellungen.
- [x] **Fortschritt für Prepare Sequence:** Pufferfüllstand, Inferenzabschnitte, Ausrichtung und globaler Tiefenbereich; Stereo zählt Frames separat. Gesamtframezahl wird von VHS übernommen, wenn verfügbar.
- [x] **Verzeichnislauf mit temporärem Tiefencache:** eigener Duke DA3 Directory Queue statt Loopy Dir; Quelldatei über beide Durchläufe festhalten, Kennung ableiten, erst nach erfolgreichem Stereo samt Audio Cache löschen und weiterschalten. Fehler halten die Queue an.
- [ ] **Abnahme im ComfyUI-Frontend:** Verzeichnislauf mit mehreren Videos sowie neue Fortschrittsanzeige prüfen. Einzelvideo-Durchläufe mit Small und Base wurden bereits erfolgreich geprüft.
