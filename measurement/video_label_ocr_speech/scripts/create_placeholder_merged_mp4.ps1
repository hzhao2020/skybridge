$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Out = Join-Path $Root "video\merged.mp4"
New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

$ffmpegExe = $null
$fc = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($fc) {
  $ffmpegExe = $fc.Source
}
if (-not $ffmpegExe) {
  foreach ($name in @("python", "python3", "py")) {
    $py = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $py) { continue }
    try {
      $ffmpegExe = (& $py.Source -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())").Trim()
      if ($ffmpegExe) { break }
    } catch {}
  }
}
if (-not $ffmpegExe -or -not (Test-Path $ffmpegExe)) {
  Write-Error "ffmpeg not found. Install ffmpeg or: pip install imageio-ffmpeg"
}

& $ffmpegExe -y `
  -f lavfi -i "testsrc=duration=20:size=640x480:rate=25" `
  -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=20" `
  -pix_fmt yuv420p -c:v libx264 -preset ultrafast -c:a aac -shortest `
  $Out

Write-Host "Wrote: $Out"
