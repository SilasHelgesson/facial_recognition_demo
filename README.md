# README

**English version below** 🇺🇸

## Deutsch 🇩🇪

### Projektbeschreibung

Dieses C++ Programm implementiert Gesichtserkennung mithilfe der Eigenfaces-Methode. Gesichter werden als Punkte in einem hochdimensionalen Raum betrachtet, und durch Hauptkomponentenanalyse (PCA) werden die wichtigsten Merkmale — die sogenannten Eigenfaces — extrahiert. Unbekannte Gesichter werden dann mit den Trainingsbildern verglichen und der ähnlichsten Person zugeordnet.

### Voraussetzungen

Stellen Sie sicher, dass Sie folgendes installiert haben:

- einen C++17 Compiler (z.B. MSVC, GCC, Clang)
- [OpenCV](https://opencv.org/) (4.x empfohlen)
- CMake (optional, je nach Build-System)

### Installation und Ausführung

1. Klonen Sie das Repository auf Ihren lokalen Rechner.

   ```bash
   git clone https://github.com/SilasHelgesson/facial_recognition_demo
   cd facial_recognition_demo
   ```

2. Öffnen Sie die Solution-Datei `facial_recognition.slnx` in Visual Studio und bauen Sie das Projekt.

3. Führen Sie das Programm aus.

   ```bash
   facial_recognition.exe <Pfad_zu_Trainingsdaten> <Pfad_zu_Testdaten> <Gesichter_anzeigen 0|1>
   ```

   Beispiel:

   ```bash
   facial_recognition.exe data/train/ data/test/ 1
   ```

### Dateistruktur

Die Trainingsdaten müssen folgendem Namensschema folgen:

```
train/
  01_00.png
  01_01.png
  ...
  15_09.png

test/
  Gordon Freeman.png
  Alyx Vance.png
  ...
```

---

## English 🇺🇸

### Project Description

This C++ program implements facial recognition using the Eigenfaces method. Faces are treated as points in a high-dimensional space, and through Principal Component Analysis (PCA) the most important features — the so-called eigenfaces — are extracted. Unknown faces are then compared against the training images and assigned to the most similar person.

### Prerequisites

Make sure you have the following installed:

- a C++17 compiler (e.g. MSVC, GCC, Clang)
- [OpenCV](https://opencv.org/) (4.x recommended)
- CMake (optional, depending on your build system)

### Installation and Execution

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/SilasHelgesson/facial_recognition_demo
   cd facial_recognition_demo
   ```

2. Open the solution file `facial_recognition.slnx` in Visual Studio and build the project.

3. Run the program.

   ```bash
   facial_recognition.exe <path_to_training_data> <path_to_testing_data> <display_faces 0|1>
   ```

   Example:

   ```bash
   facial_recognition.exe data/train/ data/test/ 1
   ```

### File Structure

Training data must follow this naming convention:

```
train/
  01_00.png
  01_01.png
  ...
  15_09.png

test/
  Gordon Freeman.png
  Alyx Vance.png
  ...
```

### License

This project is licensed under the GPL-3.0 License.
