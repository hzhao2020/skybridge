请将待测长视频命名为 merged.mp4 放在本目录后再打包上传 VM。

默认路径：本目录下的 merged.mp4（与 run_measurement.py 默认 --video 一致）。

若无长视频、仅需在 VM 上冒烟验证依赖与权限，可在仓库根目录执行：
  Linux/macOS: bash scripts/create_placeholder_merged_mp4.sh
  Windows:     powershell -File scripts/create_placeholder_merged_mp4.ps1

占位片约 20 秒（彩条 + 正弦音频），可用于打通上传与 VI 全流程；正式测耗时请换成真实业务视频。
