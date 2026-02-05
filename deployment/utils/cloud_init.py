from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def render_cloud_init_user_data(
    username: str,
    ssh_keys: list[str],
    root_password: str | None = None,
    template_filename: str = "cloud-init.yaml.jinja2",
) -> str:
    if not len(ssh_keys):
        raise RuntimeError("No ssh keys provided")

    template_path = (
        Path(__file__).resolve().parent.parent / "templates" / template_filename
    )

    env = Environment(
        loader=FileSystemLoader(Path(template_path).parent),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(Path(template_path).name)

    rendered = template.render(
        username=username,
        ssh_keys=ssh_keys,
        root_password=root_password.strip() if root_password else None,
    )

    return rendered


def render_cloud_init_meta_data(instance_name: str) -> str:
    rendered = (
        f"instance-id: {instance_name}\n" f"local-hostname: {instance_name}\n"
    )

    return rendered
