# Vibration Analysis & Predictive Maintenance

โปรเจกต์นี้ใช้ข้อมูล waveform ความสั่นสะเทือนที่มีจำนวนน้อยและไม่มี label เพื่อประเมินสุขภาพเครื่องจักร, ตรวจจับความผิดปกติ, และประมาณช่วงเวลาที่อาจเข้าใกล้ failure threshold โดยผสมผสาน signal processing, statistical rule, และ unsupervised anomaly detection เข้าด้วยกัน

## Pipeline

1. `core/parser.py`
   อ่านไฟล์ `.txt` จาก data logger แล้วแปลง waveform 1 ไฟล์ให้เป็น measurement-level feature vector
   ฟีเจอร์หลักที่สกัดได้:
   `Velocity_RMS`, `Acceleration_RMS_G`, `Peak_to_Peak_G`, `Kurtosis`, `Crest_Factor`, `Velocity_Peak_mm_s`, `Dominant_Frequency_Hz`, `Spectral_Centroid_Hz`, `Spectral_Entropy`

2. `core/classifier.py`
   ประเมินแต่ละ measurement ด้วย 3 ชั้นพร้อมกัน
   `ISO_Status`: เทียบค่า `Velocity_RMS` กับ threshold ตาม ISO 10816-3
   `Stat_Alert`: เทียบกับ baseline ช่วงแรกของแต่ละเครื่องด้วยกฎ `mean + 3 sigma`
   `Isolation_Anomaly`: ใช้ `IsolationForest` บน feature vector เพื่อหาจุดที่เบี่ยงจากกลุ่มข้อมูลรวม

3. `core/forecast.py`
   ฟิตเส้นแนวโน้มเชิงเส้นแบบถ่วงน้ำหนักข้อมูลล่าสุด (recency-weighted regression) ของ `Velocity_RMS` ต่อเครื่อง แล้วหาเวลาที่เส้นแนวโน้มจะตัด `LVL_CRIT`
   ถ้า slope ไม่เป็นบวกหรือข้อมูลยังไม่ชี้ว่าเสื่อมลง ระบบจะรายงาน `Stable or Improving`

4. `core/visualize.py`
   สร้างกราฟ 3 ชุด:
   `plot_1_vibration_timeline.png` : แยกกราฟรายเครื่อง, เห็นค่าจริง + trend fit + เส้น forecast + threshold
   `plot_2_feature_anomaly_map.png` : heatmap ค่าล่าสุดเทียบ reference (สีคือ ratio, ตัวเลขคือค่าจริง)
   `plot_3_anomaly_heatmap.png` : monthly scoreboard ของ severity และ alert points ต่อเครื่อง

5. `main.py`
   รวมผลทั้งหมดเป็นตารางสรุปหน้า terminal พร้อม export CSV สองชุด
   `output/vibration_analysis_summary.csv`
   `output/vibration_feature_history.csv`

## Output Columns

ไฟล์ `vibration_analysis_summary.csv` เน้นสถานะล่าสุดต่อเครื่อง:

- `Latest_Velocity_RMS`: ค่า velocity RMS ล่าสุด (mm/s)
- `Latest_Kurtosis`: ใช้จับแรงกระแทกหรือ waveform ที่โด่งผิดปกติ
- `Latest_Crest_Factor`: peak/RMS สำหรับชี้ early bearing or gear fault
- `ISO_Status`: ระดับความรุนแรงตามมาตรฐาน ISO
- `Health_Flag`: สรุประดับความเสี่ยงสุดท้าย (`Normal`, `Watchlist`, `High risk`, `Critical`)
- `Stat_Alert`: ผลจาก baseline 3-sigma
- `Isolation_Anomaly`: ผลจาก IsolationForest
- `Trend`: สถานะแนวโน้ม degradation
- `Est_Failure`: วันที่ประมาณว่าจะชน `LVL_CRIT` ถ้า trend ยังเป็นบวกต่อเนื่อง

ไฟล์ `vibration_feature_history.csv` เก็บ measurement history แบบละเอียด พร้อม feature และ score ทุกตัวสำหรับนำไปวิเคราะห์ต่อใน pandas, Power BI, หรือ notebook

## Setup

### 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure thresholds

```bash
cp .env.example .env
```

ตัวแปรสำคัญใน `.env`

- `LVL_WARN`, `LVL_ALERT`, `LVL_CRIT`: ISO velocity thresholds
- `BASELINE_POINTS`: จำนวน measurement ช่วงต้นที่ถือเป็น baseline ปกติของแต่ละเครื่อง
- `SIGMA_MULTIPLIER`: ตัวคูณ sigma สำหรับ statistical alert
- `FORECAST_RECENCY_WEIGHT`: น้ำหนักของจุดล่าสุดในการ fit เส้นทำนาย `Est_Failure` (ค่าสูงจะเน้นแนวโน้มล่าสุดมากขึ้น)
- `DATA_DIR`, `OUTPUT_DIR`: path ของ input/output

### 3. Run

```bash
./.venv/bin/python main.py
```

## Project Structure

```text
prediction_vibration/
├── core/
│   ├── parser.py
│   ├── classifier.py
│   ├── forecast.py
│   └── visualize.py
├── data/
├── output/
├── main.py
├── requirements.txt
└── .env.example
```

## Practical Notes

- ข้อมูลน้อยมากยังเหมาะกับแนวทางนี้ เพราะแต่ละ feature มีความหมายทางวิศวกรรมและตีความหน้างานได้
- `IsolationForest` ใช้เป็นสัญญาณเสริม ไม่ควรใช้แทน judgement ทางวิศวกรรมเพียงอย่างเดียวเมื่อ sample ยังน้อย
- ถ้ามีข้อมูลรอบหมุน (RPM), bearing fault frequencies, หรือ load condition เพิ่มเข้ามา สามารถต่อยอด frequency-domain diagnosis ให้ชี้ root cause ได้ละเอียดขึ้น
