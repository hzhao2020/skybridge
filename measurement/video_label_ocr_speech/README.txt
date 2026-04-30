自包含测量包（内含 ops/、scripts/、video/、results/），可整体打包上传到 GCP us-west1 VM。

打包前检查清单：

1. 换行符：目标为 Linux VM 时，源码与 *.sh 须为 LF（本目录已提供 .gitattributes / .editorconfig；从 Windows 手工打包前可用编辑器或 git 归一化，避免 bash 报 $'\r' 或 shebang 找不到解释器）。
2. 视频：将待测长视频命名为 video/merged.mp4（或与 --video 一致的路径）。若仓库不提交大文件，打包 zip/tar 时请手动把 merged.mp4 打进同一目录。
3. 占位片：无长视频时可执行 bash scripts/create_placeholder_merged_mp4.sh（Linux VM）或 powershell -File scripts/create_placeholder_merged_mp4.ps1（Windows）；仓库内已附带一份约 20 秒的 merged.mp4 供冒烟，正式测耗时请替换为业务素材。
4. 配置：复制 config_example.py 为 config.py（按需填写桶名等）；仅用实例服务账号 ADC 时常可省略 config.py。
5. 依赖：VM 上 chmod +x install_vm.sh scripts/create_placeholder_merged_mp4.sh 后执行 bash install_vm.sh；或手动 sudo apt install ffmpeg python3-pip 再 pip install -r requirements.txt。若还需 Vertex 等能力（本脚本不需要），可 pip install -r requirements_full.txt。
6. 凭证：VM 服务账号需具备 GCS 与 Video Intelligence 权限（推荐 ADC）。

与 segment_split_measurement 相同的批量规则：在本目录执行
  python batch_sweep_measurements.py
即按 2→4→6→…→30 分钟截取，每种时长重复 10 次；结果 CSV 与 JSON 在 results/。

单次冒烟可执行：
  python run_measurement.py --minutes 2

批量脚本默认不对 VI LRO 设客户端超时（无限等待 operation.result）；单次调试可加 --annotate-timeout-sec。

OCR / Speech 的 registry pid 与 VI Feature 名定义在 ops/registry.py（亦可 from pids import VI_OCR_GOOGLE_US 等）；config_example.py 中有对照说明。

timings_sec 含义见 run_measurement.py 顶部说明；CSV 列名见 batch_sweep_measurements.py 中 CSV_FIELDS。
