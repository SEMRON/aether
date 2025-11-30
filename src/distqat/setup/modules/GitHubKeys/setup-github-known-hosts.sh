#!/bin/bash
set -e
mkdir -p /opt/aether
chmod 755 /opt/aether
# Collect host keys for both HTTPS and SSH endpoints
ssh-keyscan -t rsa,ecdsa,ed25519 github.com > /opt/aether/known_hosts 2>/dev/null
ssh-keyscan -t rsa,ecdsa,ed25519 ssh.github.com >> /opt/aether/known_hosts 2>/dev/null
chmod 644 /opt/aether/known_hosts

# Add known hosts to system-wide SSH known_hosts file
if [ ! -f /etc/ssh/ssh_known_hosts ]; then
  cp /opt/aether/known_hosts /etc/ssh/ssh_known_hosts
  chmod 644 /etc/ssh/ssh_known_hosts
else
  # Add only new entries that don't already exist
  while IFS= read -r line; do
    if ! grep -Fxq "$line" /etc/ssh/ssh_known_hosts 2>/dev/null; then
      echo "$line" >> /etc/ssh/ssh_known_hosts
    fi
  done < /opt/aether/known_hosts
fi

echo "-o IdentitiesOnly=yes -o UserKnownHostsFile=/opt/aether/known_hosts -o StrictHostKeyChecking=yes" > /opt/aether/ssh_args.known_hosts
