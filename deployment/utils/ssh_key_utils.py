from pathlib import Path


def these_or_all_keys(these: list[str] | None):
    if these is None:
        return get_all_public_keys()
    return these


def get_all_public_keys() -> list[str]:
    """
    We copy all the public keys from this user
    (including the authorized keys for this user)
    into the created instances.
    """
    ssh_dir = Path.home() / ".ssh"
    if not ssh_dir.is_dir():
        return []

    pub_keys = []

    # Get individual public key files
    for key_file in ssh_dir.glob("*.pub"):
        try:
            pub_keys.append(key_file.read_text().strip())
        except:
            pass

    # Get keys from authorized_keys if it exists
    auth_keys = ssh_dir / "authorized_keys"
    if auth_keys.is_file():
        try:
            pub_keys.extend(
                [
                    k.strip()
                    for k in auth_keys.read_text().splitlines()
                    if k.strip()
                ]
            )
        except:
            pass

    return pub_keys
