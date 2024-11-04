# Чекпоинт 2. Разведочный анализ данных

## Шаг 1. Сбор данных

### Датасет

Выбранный датасет: [RSNA Pneumonia Detection](www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview).
Для удобства доступа датасет [загружен](https://drive.google.com/uc?id=1nIW5qgn4MurehHDiulrTMHNQMpsu4SRJ) на Google Drive.

Датасет содержит рентгенограммы грудной клетки человека. На части изображений легкие поражены пневмонией.

Данные были собраны и предоставлены [Radiological Society of North America (RSNA®)](http://www.rsna.org/) в сотрудничестве с [US National Institutes of Health](https://www.nih.gov/), [The Society of Thoracic Radiology](http://thoracicrad.org/), [MD.ai](https://www.md.ai/) и сообществом [Kaggle](https://www.kaggle.com/).

### Описание данных

Для каждой рентгенограммы в датасете указана целевая переменная (наличие/отсутствие пневмонии) и, в случае положительного класса, заданы границы области(областей), где обнаружены визуальные признаки заболевания.

<figure>
    <img src="https://i.ibb.co/hdkRx2c/image.png" alt="Пример рентгенограмм из датасета" width="300"/>
    <figcaption style="font-style: italic;">Пример рентгенограмм из датасета</figcaption>
</figure>

Файлы в датасете:
- Набор изображений в формате [DICOM](https://ru.wikipedia.org/wiki/DICOM)
- `train_labels.csv` - cодержит идентификаторы пациентов, целевую переменную и bounding box`ы.
- `detailed_class_info.csv` - предоставляет информацию о типе положительного или отрицательного класса для каждого изображения

#### DICOM-файлы

Каждый DICOM-файл, помимо собственно изображения, содержит некоторую информацию о пациенте, исследовании и изображении (пол, возраст, время и место исследования, характеристики цвета и сжатия изображения и т.п.) 

<details>
  <summary>Пример содержания DICOM-файла</summary>
  <pre>
Dataset.file_meta -------------------------------
(0002,0000) File Meta Information Group Length  UL: 202
(0002,0001) File Meta Information Version       OB: b'\x00\x01'
(0002,0002) Media Storage SOP Class UID         UI: Secondary Capture Image Storage
(0002,0003) Media Storage SOP Instance UID      UI: 1.2.276.0.7230010.3.1.4.8323329.28530.1517874485.775526
(0002,0010) Transfer Syntax UID                 UI: JPEG Baseline (Process 1)
(0002,0012) Implementation Class UID            UI: 1.2.276.0.7230010.3.0.3.6.0
(0002,0013) Implementation Version Name         SH: 'OFFIS_DCMTK_360'
-------------------------------------------------
(0008,0005) Specific Character Set              CS: 'ISO_IR 100'
(0008,0016) SOP Class UID                       UI: Secondary Capture Image Storage
(0008,0018) SOP Instance UID                    UI: 1.2.276.0.7230010.3.1.4.8323329.28530.1517874485.775526
(0008,0020) Study Date                          DA: '19010101'
(0008,0030) Study Time                          TM: '000000.00'
(0008,0050) Accession Number                    SH: ''
(0008,0060) Modality                            CS: 'CR'
(0008,0064) Conversion Type                     CS: 'WSD'
(0008,0090) Referring Physician's Name          PN: ''
(0008,103E) Series Description                  LO: 'view: PA'
(0010,0010) Patient's Name                      PN: '0004cfab-14fd-4e49-80ba-63a80b6bddd6'
(0010,0020) Patient ID                          LO: '0004cfab-14fd-4e49-80ba-63a80b6bddd6'
(0010,0030) Patient's Birth Date                DA: ''
(0010,0040) Patient's Sex                       CS: 'F'
(0010,1010) Patient's Age                       AS: '51'
(0018,0015) Body Part Examined                  CS: 'CHEST'
(0018,5101) View Position                       CS: 'PA'
(0020,000D) Study Instance UID                  UI: 1.2.276.0.7230010.3.1.2.8323329.28530.1517874485.775525
(0020,000E) Series Instance UID                 UI: 1.2.276.0.7230010.3.1.3.8323329.28530.1517874485.775524
(0020,0010) Study ID                            SH: ''
(0020,0011) Series Number                       IS: '1'
(0020,0013) Instance Number                     IS: '1'
(0020,0020) Patient Orientation                 CS: ''
(0028,0002) Samples per Pixel                   US: 1
(0028,0004) Photometric Interpretation          CS: 'MONOCHROME2'
(0028,0010) Rows                                US: 1024
(0028,0011) Columns                             US: 1024
(0028,0030) Pixel Spacing                       DS: [0.14300000000000002, 0.14300000000000002]
(0028,0100) Bits Allocated                      US: 8
(0028,0101) Bits Stored                         US: 8
(0028,0102) High Bit                            US: 7
(0028,0103) Pixel Representation                US: 0
(0028,2110) Lossy Image Compression             CS: '01'
(0028,2114) Lossy Image Compression Method      CS: 'ISO_10918_1'
(7FE0,0010) Pixel Data                          OB: Array of 142006 elements
</pre>
</details>

#### Таблица train_labels.csv

Содержание `train_labels.csv`:
- `patientId` - уникальный ID пациента
- `x` - x-координата левого верхнего угла  bounding box`а
- `y` - у-координата левого верхнего угла bounding box`а
- `width` - ширина bounding box`а
- `height` - высота bounding box`а
- `Target` - бинарная целевая переменная, указывающая, есть ли в данном образце признаки пневмонии.

<figure>
    <img src="https://i.ibb.co/FWkR15C/target-distribution.png" alt="Распределение бинарной целевой переменной" width="300"/>
    <figcaption style="font-style: italic;">Распределение бинарной целевой переменной</figcaption>
</figure>

#### Таблица detailed_class_info.csv

Содержание `detailed_class_info.csv`:
- `patientId` - уникальный ID пациента
- `class` - один из трех классов ([разъяснение организаторов](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/discussion/64723)):
    - `Lung Opacity` - признаки пневмонии обнаружены
    - `No Lung Opacity / Not Normal` - отсутствие непрозрачности, вызывающей подозрение на пневмонию (Могут присутствовать другие непрозрачные участки, не связанные с пневмонией)
    - `Normal` - признаков пневмонии не обнаружено

<figure>
    <img src="https://i.ibb.co/9G1zD1b/classes-distribution.png" alt="Распределение типов классов" width="300"/>
    <figcaption style="font-style: italic;">Распределение типов классов</figcaption>
</figure>
