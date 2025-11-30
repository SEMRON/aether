#!/bin/bash
set -e
# A small built-in word list (feel free to expand)
ADJECTIVES=(brave silent clever rapid bright bold calm wise eager swift)
ANIMALS=(falcon tiger fox wolf bear panther hawk lynx otter eagle)
COLORS=(crimson amber azure ivory olive indigo silver scarlet coral jade)
# Get machine-id and hash it deterministically
MID=$(cat /etc/machine-id)
HASH=$(echo -n "$MID" | sha256sum | cut -d' ' -f1)
# Convert hash into integer slices
A_INDEX=$(( 0x${HASH:0:2} % ${#ADJECTIVES[@]} ))
B_INDEX=$(( 0x${HASH:2:2} % ${#COLORS[@]} ))
C_INDEX=$(( 0x${HASH:4:2} % ${#ANIMALS[@]} ))
HOSTNAME="${ADJECTIVES[$A_INDEX]}-${COLORS[$B_INDEX]}-${ANIMALS[$C_INDEX]}"
hostnamectl set-hostname "$HOSTNAME"
echo "$HOSTNAME" > /etc/hostname
echo "127.0.1.1 $HOSTNAME" >> /etc/hosts
