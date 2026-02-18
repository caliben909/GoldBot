$path = '.git'
if (Test-Path -Path $path) {
    Write-Host 'Deleting .git directory...'
    $items = Get-ChildItem -Path $path -Recurse -Force
    foreach ($item in $items) {
        try {
            $item.Attributes = 'Normal'
        } catch {
            Write-Host "Could not change attributes for $($item.FullName): $_"
        }
    }
    try {
        Remove-Item -Recurse -Force -Path $path -ErrorAction Stop
        Write-Host '.git directory deleted successfully.'
    } catch {
        Write-Host "Error deleting .git directory: $_"
        Write-Host 'Trying with robocopy workaround...'
        $null = New-Item -ItemType Directory -Path 'empty_dir' -Force
        robocopy.exe 'empty_dir' $path /MIR /R:3 /W:2 > $null
        Remove-Item -Recurse -Force -Path $path -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force -Path 'empty_dir' -ErrorAction SilentlyContinue
    }
} else {
    Write-Host '.git directory does not exist.'
}