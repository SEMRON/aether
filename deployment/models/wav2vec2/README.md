```
python3 deploy.py deploy --setup-definition models/wav2vec2/prime-intellect.toml
python3 deploy.py status --setup-definition models/wav2vec2/prime-intellect.toml --auto-update

ANSIBLE_NOCOWS=1 ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i workdir_deploy/inventory.yaml models/wav2vec2/start_servers.yaml

python3 deploy.py destroy --setup-definition models/wav2vec2/prime-intellect.toml all
```