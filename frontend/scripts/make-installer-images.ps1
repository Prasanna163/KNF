# make-installer-images.ps1
# Generates branded installer-header.bmp (150x57) and installer-sidebar.bmp (164x314)
# from public/icon.png, using System.Drawing (no npm deps required).

Add-Type -AssemblyName System.Drawing

$publicDir = Join-Path $PSScriptRoot "..\public"
$iconPath  = Join-Path $publicDir "icon.png"
$headerOut = Join-Path $publicDir "installer-header.bmp"
$sidebarOut = Join-Path $publicDir "installer-sidebar.bmp"

if (-not (Test-Path $iconPath)) {
    Write-Error "icon.png not found at $iconPath"
    exit 1
}

$srcIcon = [System.Drawing.Image]::FromFile($iconPath)

function New-HighQualityGraphics($bmp) {
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    return $g
}

# ---------------------------------------------------------------------------
# Header: 150x57, white background, wordmark left + small icon badge right
# ---------------------------------------------------------------------------
$headerW = 150; $headerH = 57
$header = New-Object System.Drawing.Bitmap $headerW, $headerH, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$g = New-HighQualityGraphics $header
$g.Clear([System.Drawing.Color]::White)

$textBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(24, 24, 27))
$font = New-Object System.Drawing.Font "Segoe UI", 11, ([System.Drawing.FontStyle]::Bold)
$textRect = New-Object System.Drawing.RectangleF 10, 0, 90, $headerH
$sf = New-Object System.Drawing.StringFormat
$sf.LineAlignment = [System.Drawing.StringAlignment]::Center
$sf.Alignment = [System.Drawing.StringAlignment]::Near
$g.DrawString("KNF Studio", $font, $textBrush, $textRect, $sf)

$badgeSize = 40
$badgeX = $headerW - $badgeSize - 10
$badgeY = [int](($headerH - $badgeSize) / 2)
$g.DrawImage($srcIcon, $badgeX, $badgeY, $badgeSize, $badgeSize)

$g.Dispose()
$header.Save($headerOut, [System.Drawing.Imaging.ImageFormat]::Bmp)
$header.Dispose()
Write-Output "Wrote $headerOut ($headerW x $headerH)"

# ---------------------------------------------------------------------------
# Sidebar: 164x314, dark brand background, centered icon + wordmark
# ---------------------------------------------------------------------------
$sideW = 164; $sideH = 314
$sidebar = New-Object System.Drawing.Bitmap $sideW, $sideH, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$g2 = New-HighQualityGraphics $sidebar

$bgColor = [System.Drawing.Color]::FromArgb(9, 9, 11)
$g2.Clear($bgColor)

# subtle radial-ish glow band behind the icon for depth
$glowRect = New-Object System.Drawing.Rectangle (-40), 40, ($sideW + 80), 180
$glowBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    $glowRect,
    [System.Drawing.Color]::FromArgb(60, 59, 130, 246),
    [System.Drawing.Color]::FromArgb(0, 9, 9, 11),
    [System.Drawing.Drawing2D.LinearGradientMode]::Vertical
)
$g2.FillRectangle($glowBrush, $glowRect)

$iconSize = 108
$iconX = [int](($sideW - $iconSize) / 2)
$iconY = 46
$g2.DrawImage($srcIcon, $iconX, $iconY, $iconSize, $iconSize)

$wordFont = New-Object System.Drawing.Font "Segoe UI", 13, ([System.Drawing.FontStyle]::Bold)
$wordBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(250, 250, 250))
$wordRect = New-Object System.Drawing.RectangleF 0, ($iconY + $iconSize + 14), $sideW, 24
$sf2 = New-Object System.Drawing.StringFormat
$sf2.Alignment = [System.Drawing.StringAlignment]::Center
$g2.DrawString("KNF Studio", $wordFont, $wordBrush, $wordRect, $sf2)

$subFont = New-Object System.Drawing.Font "Segoe UI", 8, ([System.Drawing.FontStyle]::Regular)
$subBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(161, 161, 170))
$subRect = New-Object System.Drawing.RectangleF 12, ($iconY + $iconSize + 40), ($sideW - 24), 40
$sf3 = New-Object System.Drawing.StringFormat
$sf3.Alignment = [System.Drawing.StringAlignment]::Center
$g2.DrawString("Non-Covalent Interaction Modeling", $subFont, $subBrush, $subRect, $sf3)

$g2.Dispose()
$sidebar.Save($sidebarOut, [System.Drawing.Imaging.ImageFormat]::Bmp)
$sidebar.Dispose()
Write-Output "Wrote $sidebarOut ($sideW x $sideH)"

$srcIcon.Dispose()
