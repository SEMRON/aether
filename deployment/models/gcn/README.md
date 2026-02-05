```
python3 deploy.py deploy --setup-definition models/gcn/prime-intellect.toml
python3 deploy.py status --setup-definition models/gcn/prime-intellect.toml --auto-update

ANSIBLE_NOCOWS=1 ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i workdir_deploy/inventory.yaml models/gcn/start_servers.yaml

python3 deploy.py destroy --setup-definition models/gcn/prime-intellect.toml all
```
