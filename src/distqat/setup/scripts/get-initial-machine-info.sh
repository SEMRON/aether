#!/bin/bash

get_machine_info() {

set -euo pipefail
trap 'echo "Gathering the machine info failed"; return 1' ERR

# Function to check for public IP
check_public_ip() {
    # Try to get external IP
    external_ip=$(curl -s -4 https://ifconfig.me 2>/dev/null || curl -s -4 https://api.ipify.org 2>/dev/null || echo "")

    # Get local IPs
    local_ips=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1')

    # Check if behind NAT
    if [ -n "$external_ip" ]; then
        is_public="false"
        while IFS= read -r local_ip; do
            if [ "$local_ip" = "$external_ip" ]; then
                is_public="true"
                break
            fi
        done <<< "$local_ips"

        if [ "$is_public" = "true" ]; then
            echo '{"has_public_ip": true, "is_behind_nat": false, "external_ip": "'$external_ip'"}'
        else
            echo '{"has_public_ip": false, "is_behind_nat": true, "external_ip": "'$external_ip'"}'
        fi
    else
        echo '{"has_public_ip": false, "is_behind_nat": "unknown", "external_ip": null}'
    fi
}

# Read /etc/os-release
if [ -f /etc/os-release ]; then
    . /etc/os-release
    os_id="${ID:-null}"
    os_version_id="${VERSION_ID:-null}"
    os_id_like="${ID_LIKE:-null}"
    platform_id="${PLATFORM_ID:-null}"
else
    os_id="null"
    os_version_id="null"
    os_id_like="null"
    platform_id="null"
fi

# Get PCI devices
pci_devices="["
first_device=true
for d in /sys/bus/pci/devices/*; do
  [[ -e "$d/class" ]] || continue
  cls=$(<"$d/class")                # e.g. 0x030000, 0x030200, 0x038000
  [[ ${cls:2:2} == "03" ]] || continue   # only Display controllers
  ven=$(<"$d/vendor")               # 0x10de NVIDIA, 0x1002 AMD/ATI
  [[ "$ven" == "0x10de" || "$ven" == "0x1002" ]] || continue
  dev=$(<"$d/device")
  addr=${d##*/}; addr=${addr#0000:}
  # Pretty class tag
  case ${cls:4:2} in
    00) cl="VGA" ;;
    02) cl="3D" ;;
    80) cl="Disp" ;;   # other display controller
    *)  cl="03??" ;;
  esac

  # Add comma if not first device
  if [ "$first_device" = true ]; then
    first_device=false
  else
    pci_devices+=","
  fi

  # Append JSON object for this device
  pci_devices+="{\"address\":\"$addr\",\"class\":\"$cl\",\"vendor\":\"${ven#0x}\",\"device\":\"${dev#0x}\"}"
done
pci_devices+="]"

# Get driver info
# Check for NVIDIA CUDA drivers
nvidia_version="null"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=driver_version --format=csv,noheader &>/dev/null
    if [ $? -eq 0 ]; then
        nvidia_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
        if [ -n "$nvidia_version" ]; then
            nvidia_version="\"$nvidia_version\""
        else
            nvidia_version="null"
        fi
    else
        nvidia_version="null"
    fi
fi

# Check for CUDA version
cuda_version="null"
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' | head -n1)
    if [ -n "$cuda_version" ]; then
        cuda_version="\"$cuda_version\""
    else
        cuda_version="null"
    fi
elif command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' | head -n1)
    if [ -n "$cuda_version" ]; then
        cuda_version="\"$cuda_version\""
    else
        cuda_version="null"
    fi
fi

# Check for AMD GPU driver version
amdgpu_version="null"
if [ -d /sys/module/amdgpu ]; then
    # Check if driver version is available in module info
    if [ -f /sys/module/amdgpu/version ]; then
        amdgpu_version=$(cat /sys/module/amdgpu/version 2>/dev/null)
        if [ -n "$amdgpu_version" ]; then
            amdgpu_version="\"$amdgpu_version\""
        else
            amdgpu_version="null"
        fi
    fi
fi

# Check for AMD ROCm drivers
rocm_version="null"
if [ -f /opt/rocm/.info/version ]; then
    rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null)
    if [ -n "$rocm_version" ]; then
        rocm_version="\"$rocm_version\""
    else
        rocm_version="null"
    fi
elif command -v rocm-smi &> /dev/null; then
    rocm_version=$(rocm-smi --version 2>/dev/null | grep -oP 'ROCm Kernel Version: \K[0-9.]+' | head -n1)
    if [ -n "$rocm_version" ]; then
        rocm_version="\"$rocm_version\""
    else
        rocm_version="null"
    fi
fi

driver_info="{\"nvidia\": {\"driver_version\": $nvidia_version, \"cuda_version\": $cuda_version}, \"amd\": {\"driver_version\": $amdgpu_version, \"rocm_version\": $rocm_version}}"

# Get network info
network_info=$(check_public_ip)

# Get user info
# Get current user
current_user=$(whoami)

# Check for sudo rights
has_sudo="false"
if { sudo -n true; } 2>/dev/null; then
    has_sudo="true"
elif groups | grep -qE '(sudo|wheel|admin)'; then
    # User is in sudo/wheel/admin group but might need password
    has_sudo="possible"
fi

user_info="{\"username\": \"$current_user\", \"has_sudo\": \"$has_sudo\"}"

# Get CPU info
cpu_arch=$(uname -m)

# Get CPU vendor from /proc/cpuinfo
cpu_vendor=$(grep -m1 'vendor_id' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//' | sed 's/"/\\"/g')
if [ -z "$cpu_vendor" ]; then
    cpu_vendor="null"
else
    cpu_vendor="$cpu_vendor"
fi

# Get CPU model name
cpu_model=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ *//' | sed 's/"/\\"/g')
if [ -z "$cpu_model" ]; then
    cpu_model="null"
else
    cpu_model="\"$cpu_model\""
fi

# Get CPU flags for instruction extensions
cpu_flags=""
if [ -f /proc/cpuinfo ]; then
    flags_line=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | cut -d: -f2)
    if [ -n "$flags_line" ]; then
        # Keep flags as a single string
        extensions="\"$flags_line\""
    else
        extensions="null"
    fi
fi

# Get CPU core count
cpu_cores=$(nproc 2>/dev/null || echo "null")

# Build CPU info JSON
cpu_info="{\"architecture\": \"$cpu_arch\", \"vendor\": \"$cpu_vendor\", \"model\": $cpu_model, \"cores\": $cpu_cores, \"extensions\": $extensions}"

# Get total memory in bytes
total_memory=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
if [ -n "$total_memory" ]; then
    # Convert from KB to bytes
    total_memory=$((total_memory * 1024))
else
    total_memory="null"
fi

# Get disk space info
# Get root filesystem disk space
root_disk_info=$(df -B1 / 2>/dev/null | tail -1)
if [ -n "$root_disk_info" ]; then
    root_total=$(echo "$root_disk_info" | awk '{print $2}')
    root_used=$(echo "$root_disk_info" | awk '{print $3}')
    root_free=$(echo "$root_disk_info" | awk '{print $4}')
else
    root_total="null"
    root_used="null"
    root_free="null"
fi

# Get user home directory disk space
user_home_dir=$(eval echo ~$current_user)
user_disk_info=$(df -B1 "$user_home_dir" 2>/dev/null | tail -1)

# Check if home directory is on the same filesystem as root
root_device=$(df / 2>/dev/null | tail -1 | awk '{print $1}')
home_device=$(df "$user_home_dir" 2>/dev/null | tail -1 | awk '{print $1}')

if [ "$root_device" != "$home_device" ] && [ -n "$user_disk_info" ]; then
    # Home is on a different filesystem, include its info
    user_total=$(echo "$user_disk_info" | awk '{print $2}')
    user_used=$(echo "$user_disk_info" | awk '{print $3}')
    user_free=$(echo "$user_disk_info" | awk '{print $4}')
    disk_info="{\"root\": {\"total\": $root_total, \"used\": $root_used, \"free\": $root_free}, \"$current_user\": {\"total\": $user_total, \"used\": $user_used, \"free\": $user_free}}"
else
    # Home is on the same filesystem as root, only include root info
    disk_info="{\"root\": {\"total\": $root_total, \"used\": $root_used, \"free\": $root_free}}"
fi

# Check for Secure Boot status
secure_boot_enabled="unknown"
if [ -d /sys/firmware/efi ]; then
    # System is UEFI, check for Secure Boot
    if [ -f /sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c ]; then
        # Read the SecureBoot variable (last byte indicates status)
        sb_value=$(od -An -t u1 -N1 -j4 /sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c 2>/dev/null | tr -d ' ')
        if [ "$sb_value" = "1" ]; then
            secure_boot_enabled="true"
        elif [ "$sb_value" = "0" ]; then
            secure_boot_enabled="false"
        fi
    elif [ -f /sys/firmware/efi/runtime-map/SecureBoot/data ]; then
        # Alternative location for SecureBoot status
        sb_value=$(od -An -t u1 -N1 /sys/firmware/efi/runtime-map/SecureBoot/data 2>/dev/null | tr -d ' ')
        if [ "$sb_value" = "1" ]; then
            secure_boot_enabled="true"
        elif [ "$sb_value" = "0" ]; then
            secure_boot_enabled="false"
        fi
    elif command -v mokutil &> /dev/null; then
        # Use mokutil if available
        mokutil_output=$(mokutil --sb-state 2>/dev/null || true)
        if echo "$mokutil_output" | grep -q "SecureBoot enabled"; then
            secure_boot_enabled="true"
        elif echo "$mokutil_output" | grep -q "SecureBoot disabled"; then
            secure_boot_enabled="false"
        fi
    fi
else
    # System is BIOS/Legacy boot, no Secure Boot
    secure_boot_enabled="false"
fi

# Create final JSON output
# Convert "null" strings to JSON null
if [ "$os_id" = "null" ]; then
    os_id_json="null"
else
    os_id_json="\"$os_id\""
fi

if [ "$os_version_id" = "null" ]; then
    os_version_id_json="null"
else
    os_version_id_json="\"$os_version_id\""
fi

if [ "$os_id_like" = "null" ]; then
    os_id_like_json="null"
else
    os_id_like_json="\"$os_id_like\""
fi

if [ "$platform_id" = "null" ]; then
    platform_id_json="null"
else
    platform_id_json="\"$platform_id\""
fi

# Construct JSON manually
output="{
    \"os\": {
        \"id\": $os_id_json,
        \"version_id\": $os_version_id_json,
        \"id_like\": $os_id_like_json,
        \"platform_id\": $platform_id_json
    },
    \"cpu\": $cpu_info,
    \"memory\": {
        \"total\": $total_memory
    },
    \"disk_info\": $disk_info,
    \"pci_devices\": $pci_devices,
    \"driver_info\": $driver_info,
    \"secure_boot_enabled\": \"$secure_boot_enabled\",
    \"network\": $network_info,
    \"user_info\": $user_info
}"

echo $output
} # end of wrapper function

# Check if running in interactive terminal (not piped)
output=$(get_machine_info)
if [ $? -eq 0 ]; then
    if [ -t 1 ]; then
        echo -e "\n\n\n#### COPY FROM BELOW THIS LINE ####\n\n\n$output\n\n"
    else
        echo "$output"
    fi
else
    echo "Gathering the machine info failed" >&2
    if [ ! -t 1 ]; then
        exit 1
    fi
fi
