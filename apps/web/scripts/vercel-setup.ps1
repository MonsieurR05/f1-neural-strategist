# Link apps/web to a Vercel project and point it at the deployed API (production env).
# Usage, from apps/web:  & .\scripts\vercel-setup.ps1 -ApiUrl https://f1-neural-strategist-api.onrender.com
param(
    [Parameter(Mandatory = $true)][string]$ApiUrl,
    [string]$Project = "f1-neural-strategist"
)
# Windows PowerShell treats a native command's stderr (Vercel's banner) as an error under
# "Stop", so check exit codes instead.
$ErrorActionPreference = "Continue"

vercel link --yes --project $Project
if ($LASTEXITCODE -ne 0) { throw "vercel link failed" }
foreach ($name in @("API_URL", "NEXT_PUBLIC_API_URL")) {
    vercel env rm $name production --yes *> $null   # absent on first run: fine
    $ApiUrl | vercel env add $name production
    if ($LASTEXITCODE -ne 0) { throw "vercel env add $name failed" }
}
vercel env ls production
