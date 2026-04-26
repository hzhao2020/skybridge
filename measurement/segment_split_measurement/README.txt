自包含测量包（内含 ops/），可整体打包上传到 VM。

1. cp config_example.py config.py 并填写 GCP、VIDEO_SPLIT_URLS等。
2. 长视频放到 video/merged.mp4
3. pip install -r requirements.txt
4. 设置 GCP 凭证（推荐 VM 服务账号 ADC，或 gcloud auth application-default login）
5. 在本目录执行：
   python run_measurement.py --minutes 2
   python batch_sweep_measurements.py

默认使用 seg_google_us + split_google_us，结果在 results/

VM 建议与 GCS 同区（如 us-west1）。若需代理，在 config.py 中设 HTTPS_PROXY 并在运行前 export https_proxy=...
