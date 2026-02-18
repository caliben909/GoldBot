
try {
    $path = ".git"
    if (Test-Path -Path $path) {
        Write-Host "Deleting directory: $path"
        # Use .NET API to delete directory with force
        [System.IO.Directory]::Delete($path, $true)
        Write-Host "Directory deleted successfully"
    } else {
        Write-Host "Directory not found: $path"
    }
} catch {
    Write-Host "Error deleting directory: $($_.Exception.Message)"
    Write-Host "Trying to kill any processes that might be locking the directory"
    
    # Try to kill any git or related processes
    try {
        Get-Process -Name git -ErrorAction SilentlyContinue | Stop-Process -Force
        Get-Process -Name code -ErrorAction SilentlyContinue | Stop-Process -Force
        Get-Process -Name explorer -ErrorAction SilentlyContinue | Stop-Process -Force
    } catch {
        Write-Host "Error killing processes: $($_.Exception.Message)"
    }
    
    # Wait a moment
    Start-Sleep -Seconds 1
    
    # Try one more time
    try {
        [System.IO.Directory]::Delete($path, $true)
        Write-Host "Directory deleted successfully after killing processes"
    } catch {
        Write-Host "Failed to delete directory after killing processes: $($_.Exception.Message)"
    }
}

# Restart explorer
try {
    Start-Process explorer
} catch {
    Write-Host "Error restarting explorer: $($_.Exception.Message)"
}
