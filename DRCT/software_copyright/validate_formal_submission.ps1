# 使用本机Word对桌面正式提交版进行分页和源程序行数校验。
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ErrorActionPreference = 'Stop'
$dir = Join-Path ([Environment]::GetFolderPath('Desktop')) '软著材料_基于DRCT的风云卫星晴空指数超分辨率重建软件V1.0\正式提交版'
$manual = Join-Path $dir '软件说明书_基于DRCT的风云卫星晴空指数超分辨率重建软件V1.0.docx'
$source = Join-Path $dir '源程序鉴别材料_基于DRCT的风云卫星晴空指数超分辨率重建软件V1.0.docx'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $manualDoc = $word.Documents.Open($manual, $false, $true)
    $manualDoc.Repaginate()
    $manualPages = $manualDoc.ComputeStatistics(2)
    $manualLinesByPage = @{}
    foreach ($p in $manualDoc.Paragraphs) {
        $page = $p.Range.Information(3) # wdActiveEndPageNumber
        $lines = $p.Range.ComputeStatistics(1) # wdStatisticLines
        if (-not $manualLinesByPage.ContainsKey($page)) { $manualLinesByPage[$page] = 0 }
        $manualLinesByPage[$page] += $lines
    }
    $manualLineRows = @($manualLinesByPage.GetEnumerator() | Sort-Object Key | ForEach-Object { [pscustomobject]@{Page=[int]$_.Key; Lines=[int]$_.Value} })
    $manualDoc.Close($false)

    $sourceDoc = $word.Documents.Open($source, $false, $true)
    $sourceDoc.Repaginate()
    $sourcePages = $sourceDoc.ComputeStatistics(2)
    $codeRows = @()
    $wrapped = @()
    foreach ($p in $sourceDoc.Paragraphs) {
        $text = $p.Range.Text.TrimEnd([char]13, [char]7)
        if ($text -match '^\d{4}:') {
            $codeRows += $text
            $lineCount = $p.Range.ComputeStatistics(1)
            if ($lineCount -gt 1) { $wrapped += [pscustomobject]@{Line=$text.Substring(0,[Math]::Min(60,$text.Length)); VisualLines=$lineCount} }
        }
    }
    $sourceDoc.Close($false)
    $summary = [pscustomobject]@{
        ManualPages = $manualPages
        ManualLinesByPage = $manualLineRows
        SourcePages = $sourcePages
        CodeRows = $codeRows.Count
        FirstCodeRow = $codeRows[0]
        LastCodeRow = $codeRows[-1]
        WrappedCodeRows = $wrapped.Count
        WrappedPreview = ($wrapped | Select-Object -First 3 | ConvertTo-Json -Compress)
    }
    $summary | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $dir 'word_validation.json') -Encoding UTF8
    $summary | ConvertTo-Json -Depth 3
}
finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
