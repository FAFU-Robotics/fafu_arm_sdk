param(
    [ValidateSet("Release", "Debug", "RelWithDebInfo", "MinSizeRel")]
    [string]$Configuration = "Release",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$pythonExe = & $Python -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$pybindDir = & $pythonExe -m pybind11 --cmakedir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$abi = & $pythonExe -c "import sys; print('cp' + str(sys.version_info.major) + str(sys.version_info.minor))"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$buildDir = Join-Path $PSScriptRoot "build-$abi"

Write-Host "Building fafu_motor for $abi with $pythonExe"
& cmake -S $PSScriptRoot -B $buildDir -G "Visual Studio 17 2022" -A x64 `
    "-Dpybind11_DIR=$pybindDir" "-DPython3_EXECUTABLE=$pythonExe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& cmake --build $buildDir --config $Configuration --parallel 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location (Join-Path $PSScriptRoot "..\fafu_robot_python")
try {
    & $pythonExe -c "import fafu_motor; import fafu_robot_controller; print('OK', fafu_motor.__file__)"
    $importResult = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $importResult
