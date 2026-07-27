<#
.SYNOPSIS
    Add or remove a pseudo (secondary/alias) IPv4 address on a local adapter,
    so you can stream to a second "machine" address without extra hardware.

.EXAMPLE
    # run PowerShell as Administrator
    .\add_pseudo_ip.ps1 -IPAddress 10.10.10.10 -InterfaceAlias 'Ethernet'
    .\add_pseudo_ip.ps1 -IPAddress 10.10.10.10 -Remove
#>
param(
    [string]$IPAddress = '10.10.10.10',
    [int]$PrefixLength = 24,
    [string]$InterfaceAlias = 'Ethernet',
    [switch]$Remove
)

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error 'This script must be run from an elevated PowerShell session.'
    exit 1
}

if ($Remove) {
    Remove-NetIPAddress -IPAddress $IPAddress -Confirm:$false
    Write-Host "removed $IPAddress"
    exit 0
}

if (-not (Get-NetAdapter -Name $InterfaceAlias -ErrorAction SilentlyContinue)) {
    Write-Error "no adapter named '$InterfaceAlias'. Available:"
    Get-NetAdapter | Select-Object Name, Status, LinkSpeed | Format-Table -AutoSize
    exit 1
}

# -SkipAsSource keeps Windows from picking this alias as the default source IP
New-NetIPAddress -InterfaceAlias $InterfaceAlias -IPAddress $IPAddress `
    -PrefixLength $PrefixLength -SkipAsSource $true | Out-Null

Write-Host "added pseudo IP $IPAddress/$PrefixLength on '$InterfaceAlias'"
Write-Host "stream to it with:  python stream_udp.py --target ${IPAddress}:5600"
Write-Host "receive with:       python recv_udp.py --bind $IPAddress --port 5600"
