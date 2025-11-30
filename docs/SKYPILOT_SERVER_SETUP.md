# Skypilot server setup template

[Skypilot](https://docs.skypilot.co/en/latest/docs/index.html) is cluster management system, which makes it easy to deploy workloads to a host of platforms.

Skypilot is originally intended mostly for managing heterogenous clusters, but we simply use it as a simple to set up and convenient way to deploy to a host of different clouds for now.

## Skypilot setup

Settin up skypilot can be quite involved, for our purposes, you can simply start by instantiating a server in an environment of your choosing, with the following cloud-init template.

With this template, you only need to insert your public keys, and then you can connect to it through ssh.

Connect to the `skyserver` user, and activate the conda environment `sky` to set up the cloud providers.
(see the official [skypilot installation docs](https://docs.skypilot.co/en/latest/getting-started/installation.html#aws) — note, you only need to regard the individual cloud setup parts, the initial install is already handled)

```bash
skyserver@1-1-1-1 :~$ source miniconda3/etc/profile.d/conda.sh
skyserver@1-1-1-1 :~$ conda activate sky

# aws setup:
skyserver@1-1-1-1 :~$ aws configure

# for all cloud platforms see the skypilot install docs
```

### Using skypilot

For using skypilot, you can connect to the `regularuser` account.

Skypilot provides api enpoints and a web interface, which are exposed on port :8080.
Connect through ssh with a port forward to be able to access the web interface locally.

`ssh -L 8080:localhost:8080 <server address>`

The dashboard is now accessible under [http://localhost:8080/dashboard](http://localhost:8080/dashboard).

The skypilot exe is provided for your convenience in `~/bin/sky`

```bash
# test api connection
skyserver@1-1-1-1 :~$ bin/sky status
# list all available gpu types
skyserver@1-1-1-1 :~$ bin/sky show-gpus -a
```

For actual development work, you will likely want to install skypilot locally. In this case, you'll have to use the [skypilot api connect command](https://docs.skypilot.co/en/latest/reference/api-server/api-server.html#sky-api-server-connect). (enter the `http://localhost:8080` endpoint when prompted)

### cloud-init template

Insert your public keys, and run on a cloud of your choosing. (taking care to keep the ports open)

```yaml
#cloud-config
package_update: true
package_upgrade: false

packages:
  - curl
  - wget
  - git
  - ca-certificates
  - apt-transport-https
  - docker.io
  - tzdata
  - unzip

users:
  - name: skyserver
    gecos: "SkyPilot API Server"
    shell: /bin/bash
    groups: [docker]
    sudo: ["ALL=(ALL) NOPASSWD:ALL"]
    ssh_authorized_keys:
      - ssh-rsa AAAAB.... your authorized key here
    lock_passwd: true
  - name: regularuser
    gecos: "Regular User"
    shell: /bin/bash
    groups: []
    sudo: []
    ssh_authorized_keys:
      - ssh-rsa AAAAB.... your authorized key here
    lock_passwd: false

write_files:
  # Miniconda (user-local) installer for 'skyserver'
  - path: /usr/local/sbin/install_miniconda_user.sh
    permissions: '0755'
    owner: root:root
    content: |
      #!/usr/bin/env bash
      set -euo pipefail
      USERHOME="$(readlink -f ~)"
      TMPDIR="$(mktemp -d)"
      trap 'rm -rf "$TMPDIR"' EXIT
      chmod 755 "$TMPDIR"
      cd "$TMPDIR"
      wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$TMPDIR/miniconda.sh"
      chmod +x "$TMPDIR/miniconda.sh"
      bash "$TMPDIR/miniconda.sh" -b -p "$USERHOME/miniconda3"
      # shell init
      echo '. "$HOME/miniconda3/etc/profile.d/conda.sh" >/dev/null 2>&1 || true' | tee -a "$USERHOME/.bashrc" >/dev/null
      echo 'export PATH="$HOME/miniconda3/bin:$PATH"' | tee -a "$USERHOME/.bashrc" >/dev/null

  # Bootstrap SkyPilot env for the server user (Python 3.10 + cloud CLIs/libs)
  - path: /usr/local/sbin/bootstrap_sky_env.sh
    permissions: '0755'
    owner: root:root
    content: |
      #!/usr/bin/env bash
      set -euo pipefail
      USERNAME="$(whoami)"
      USERHOME="$(readlink -f ~)"
      export HOME="$USERHOME"
      export PATH="$USERHOME/miniconda3/bin:$PATH"
      . "$USERHOME/miniconda3/etc/profile.d/conda.sh"
      conda config --set always_yes yes --set changeps1 no
      conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
      conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
      if ! conda env list | grep -qE '^\s*sky\s'; then
        conda create -n sky python=3.10
      fi
      # Install SkyPilot client bits for cloud logins/checks
      conda run -n sky python -m pip install --upgrade pip
      conda run -n sky python -m pip install "skypilot[all]"
      # Convenience shims
      install -d -m 0755 -o "$USERNAME" -g "$USERNAME" "$USERHOME/bin"
      ln -sf "$USERHOME/miniconda3/envs/sky/bin/sky" "$USERHOME/bin/sky"
      ln -sf "$USERHOME/miniconda3/envs/sky/bin/pip" "$USERHOME/bin/pip-sky"

  # Systemd unit to manage the API server as 'skyserver'
  - path: /etc/systemd/system/skypilot-api-server.service
    permissions: '0644'
    owner: root:root
    content: |
      [Unit]
      Description=SkyPilot API Server (Docker, loopback-only)
      After=docker.service network-online.target
      Wants=docker.service network-online.target

      [Service]
      Type=simple
      User=skyserver
      Group=docker
      WorkingDirectory=/home/skyserver
      ExecStart=/usr/bin/docker run --rm --name skypilot-api-server -p 127.0.0.1:8080:46580 -v /home/skyserver:/root -e SKYPILOT_GLOBAL_CONFIG=/root/.api-server-config.yaml --entrypoint tini berkeleyskypilot/skypilot:0.10.5 -- bash -lc 'pip install msrestazure && sky api start --deploy --foreground'
      ExecStop=/usr/bin/docker stop skypilot-api-server
      Restart=always
      RestartSec=5
      # Hardening
      NoNewPrivileges=yes
      PrivateTmp=yes
      ProtectHome=no
      ProtectSystem=full
      LockPersonality=yes
      CapabilityBoundingSet=
      RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
      RestrictNamespaces=yes
      RestrictRealtime=yes

      [Install]
      WantedBy=multi-user.target

  - path: /usr/local/sbin/harden_ufw_local8080.sh
    permissions: '0755'
    owner: root:root
    content: |
      #!/usr/bin/env bash
      set -euo pipefail
      if command -v ufw >/dev/null 2>&1; then
        ufw allow OpenSSH || true
        # Deny 8080 from non-local addresses
        ufw deny in to any port 8080 proto tcp || true
        yes | ufw enable || true
      fi

runcmd:
  # Enable and start Docker
  - [systemctl, enable, --now, docker]
  # Create and provision the 'skyserver' user's conda environment
  - [sudo, -u, skyserver, /usr/local/sbin/install_miniconda_user.sh]
  - [sudo, -u, skyserver, /usr/local/sbin/bootstrap_sky_env.sh]
  # Ensure docker group permissions for the user are active for services
  - [usermod, -aG, docker, skyserver]
  # place dummy api server config
  - [sudo, -u, skyserver, touch, /home/skyserver/.api-server-config.yaml]
  # Start API server
  - [systemctl, enable, --now, skypilot-api-server.service]
  # apply UFW hardening
  - [/usr/local/sbin/harden_ufw_local8080.sh]
  # Install skypilot for the user
  - [sudo, -u, regularuser, /usr/local/sbin/install_miniconda_user.sh]
  - [sudo, -u, regularuser, /usr/local/sbin/bootstrap_sky_env.sh]
  # Create .sky directory and config.yaml for regularuser
  - [sudo, -u, regularuser, bash, -c, 'mkdir -p /home/regularuser/.sky && echo "api_server:" > /home/regularuser/.sky/config.yaml && echo "  endpoint: http://localhost:8080" >> /home/regularuser/.sky/config.yaml']

final_message: |
  SkyPilot API server deployed.
  - Container binds to 127.0.0.1:8080 only.
  - Service: systemctl status skypilot-api-server.service
  - Server user env: sudo -iu skyserver "source ~/miniconda3/etc/profile.d/conda.sh && conda activate sky"
  - Configure cloud creds as 'skyserver':
      activate skypilot conda env:
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate sky
      and then configure the individual clouds according to the skypilot docs:
        https://docs.skypilot.co/en/latest/getting-started/installation.html
  - To access the dashboard: create an SSH tunnel to this host and open http://localhost:8080/dashboard
    ssh tunnel command:
      ssh -L 8080:localhost:8080 regularuser@<server_ip>

```
