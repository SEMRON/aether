#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Ansible task files via a wrapper playbook."
    )
    parser.add_argument(
        "--inventory",
        "-i",
        default="workdir_deploy/inventory.yaml",
        help="Inventory file to use",
    )
    parser.add_argument(
        "--vars-files", "-f", nargs="*", help="YAML variable files to include"
    )
    parser.add_argument(
        "--define", "-D", action="append", help="Extra variables in key=value format (can be used multiple times)"
    )
    parser.add_argument(
        "--hosts", default="all", help="Target hosts (default: all)"
    )
    parser.add_argument(
        "--prefix-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ansible-tasks/"
        ),
        help="Directory prefix for task files",
    )
    parser.add_argument(
        "task_files", nargs="+", help="Task file names or paths to include"
    )
    return parser.parse_args()


def load_vars_files(var_files):
    combined_vars = {}
    for vf in var_files:
        with open(vf, "r") as f:
            data = yaml.safe_load(f)
            if data:
                combined_vars.update(data)
    return combined_vars


def parse_define_vars(defines):
    result = {}
    for d in defines:
        if "=" not in d:
            continue
        k, v = d.split("=", 1)
        result[k] = v
    return result


def resolve_task_paths(task_files, prefix_dir):
    resolved = []
    for tf in task_files:
        # Make sure filename ends with .yaml
        if not tf.endswith(".yaml"):
            tf = tf + ".yaml"

        if os.path.isabs(tf) or os.path.exists(tf):
            resolved.append(tf)
        else:
            resolved.append(os.path.join(prefix_dir, tf))
    return resolved


def prepare_ansible_command(
    inventory: str,
    hosts: str,
    task_files: list[str],
    prefix_dir: str,
    vars_files: list[str] = list(),
    define: list[str] = list(),
):
    extra_vars = {
        "target_hosts": hosts,
        "task_files": resolve_task_paths(task_files, prefix_dir),
    }

    if vars_files:
        file_vars = load_vars_files(vars_files)
        extra_vars.update(file_vars)

    if define:
        cli_vars = parse_define_vars(define)
        extra_vars.update(cli_vars)

    extra_vars_arg = yaml.safe_dump(extra_vars, default_flow_style=True)

    print(f"Extra vars: {extra_vars_arg}")

    cmd = [
        "ansible-playbook",
        "-i",
        Path(inventory).absolute(),
        Path(os.path.abspath(__file__)).parent.absolute()
        / "templates"
        / "wrapper-playbook.yaml",
        "--extra-vars",
        extra_vars_arg,
    ]

    return cmd


def run_tasks(
    inventory: str,
    hosts: str,
    task_files: list[str],
    prefix_dir: str,
    vars_files: list[str] = list(),
    define: list[str] = list(),
):
    cmd = prepare_ansible_command(
        inventory=inventory,
        hosts=hosts,
        task_files=task_files,
        prefix_dir=prefix_dir,
        vars_files=vars_files,
        define=define,
    )
    run_ansible_command(cmd)


def run_ansible_command(cmd):
    print("Executing command: ", " ".join([str(arg) for arg in cmd]))

    # Set environment variables for Ansible
    env = os.environ.copy()
    env["ANSIBLE_HOST_KEY_CHECKING"] = "False"
    env["ANSIBLE_NOCOWS"] = "1"

    # Change to the directory where this script is located
    script_dir = Path(os.path.abspath(__file__)).parent.absolute()
    # Run the subprocess from the script directory
    subprocess.run(
        cmd, stdout=sys.stdout, stderr=sys.stderr, env=env, cwd=script_dir
    )


def main():
    args = parse_args()

    run_tasks(
        inventory=args.inventory,
        hosts=args.hosts,
        task_files=args.task_files,
        prefix_dir=args.prefix_dir,
        vars_files=args.vars_files or [],
        define=args.define or [],
    )


if __name__ == "__main__":
    main()
