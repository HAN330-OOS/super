# 使用本机 Word 打开生成的DOCX并进行分页统计；不修改正文。
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ErrorActionPreference = 'Stop'
$out = Join-Path $PSScriptRoot 'output'
$files = Get-ChildItem -LiteralPath $out -Filter '*.docx' | Sort-Object Name
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($file in $files) {
        $doc = $word.Documents.Open($file.FullName, $false, $true)
        $doc.Repaginate()
        $pages = $doc.ComputeStatistics(2) # wdStatisticPages
        $words = $doc.ComputeStatistics(0) # wdStatisticWords
        [pscustomobject]@{File=$file.Name; Pages=$pages; Words=$words; Size=$file.Length}
        $doc.Close($false)
    }
}
finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
